import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
from .rl_utils import *
import sys
import random
from PIL import Image, ImageDraw, ImageFont
import torchvision
import requests
import io
from torchvision.utils import save_image
import os
from datetime import datetime
totensor = torchvision.transforms.ToTensor()
from diffusers import AutoencoderTiny
from algorithms.pipeline_rf import *
import matplotlib.pyplot as plt

def obs_to_tensors(obs,size):
    flatten_obs = obs.view(-1, 3, 84, 84)
    flatten_obs = F.interpolate(flatten_obs, size = (size,size), mode='bilinear', align_corners=False)
    return flatten_obs/255.0

def get_background_image_tensor(initial_latent, transform_strength, port):
    # Serialize initial_latent into an in-memory file
    buffer_initial = io.BytesIO()
    torch.save(initial_latent.cpu(), buffer_initial)
    buffer_initial.seek(0)

    # Build the request URL and payload
    url = "http://localhost:{}/generate".format(port)
    files = {
        'initial_latent': ('initial_latent.pt', buffer_initial, 'application/x-pytorch')
    }
    data = {
        'transform_strength': transform_strength  # sent as form data
    }

    # Send the request
    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        response_buffer = io.BytesIO(response.content)
        generated_images = torch.load(response_buffer)
        return generated_images
    else:
        raise ValueError(f"Server returned an error: {response.status_code}, {response.text}")



class SAC(object):
	def __init__(self, obs_shape, action_shape, args):
		self.args=args
		self.device=torch.device("cuda:{}".format(args.gpu))
		self.discount = args.discount
		self.critic_tau = args.critic_tau
		self.encoder_tau = args.encoder_tau
		self.actor_update_freq = args.actor_update_freq
		self.critic_target_update_freq = args.critic_target_update_freq
		self.args=args
		shared_cnn = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters).to(self.device)
		head_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).to(self.device)
		actor_encoder = m.Encoder(
			shared_cnn,
			head_cnn,
			m.RLProjection(head_cnn.out_shape, args.projection_dim)
		)
		critic_encoder = m.Encoder(
			shared_cnn,
			head_cnn,
			m.RLProjection(head_cnn.out_shape, args.projection_dim)
		)
		#self.pipeline
		self.actor = m.Actor(actor_encoder, action_shape, args.hidden_dim, args.actor_log_std_min, args.actor_log_std_max).to(self.device)
		self.critic = m.Critic(critic_encoder, action_shape, args.hidden_dim).to(self.device)
		self.critic_target = deepcopy(self.critic)
		
		self.pipe =  RectifiedFlowPipeline.from_pretrained("XCLIU/instaflow_0_9B_from_sd_1_5", safety_checker=None,force_download=False,torch_dtype=torch.float16).to("cuda")
		self.pipe.vae = AutoencoderTiny.from_pretrained(
    "sayakpaul/taesd-diffusers", torch_dtype=torch.float16,
).to("cuda")

		self.log_alpha = torch.tensor(np.log(args.init_temperature)).to(self.device)
		self.log_alpha.requires_grad = True
		self.target_entropy = -np.prod(action_shape)

		self.actor_optimizer = torch.optim.Adam(
			self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
		)
		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
		)
		self.log_alpha_optimizer = torch.optim.Adam(
			[self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
		)
		self.negative_prompt_embeddings=torch.randn(1,77,768).to(self.device)
		self.prompt_embeddings=torch.randn(1,77,768).to(self.device)
		self.prompt_embeddings=240*self.prompt_embeddings/torch.norm(self.prompt_embeddings)
		self.train()
		self.critic_target.train()


	def train(self, training=True):
		self.training = training
		self.actor.train(training)
		self.critic.train(training)

	def eval(self):
		self.train(False)

	@property
	def alpha(self):
		return self.log_alpha.exp()
		
	def _obs_to_input(self, obs):
		if isinstance(obs, utils.LazyFrames):
			_obs = np.array(obs)
		else:
			_obs = obs
		_obs = torch.FloatTensor(_obs).to(self.device)
		_obs = _obs.unsqueeze(0)
		return _obs

	def select_action(self, obs):
		_obs = self._obs_to_input(obs)
		with torch.no_grad():
			mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
		return mu.cpu().data.numpy().flatten()
	def select_action2(self, obs):
		with torch.no_grad():
			mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
		return mu
	def sample_action(self, obs):
		_obs = self._obs_to_input(obs)
		with torch.no_grad():
			mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
		return pi.cpu().data.numpy().flatten()

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		current_Q1, current_Q2 = self.critic(obs, action)
		critic_loss = F.mse_loss(current_Q1,
								 target_Q) + F.mse_loss(current_Q2, target_Q)

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)


		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
		_, pi, log_pi, log_std = self.actor(obs, detach=True)
		actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

		actor_Q = torch.min(actor_Q1, actor_Q2)
		actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

		if L is not None:
			L.log('train_actor/loss', actor_loss, step)
			entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
												) + log_std.sum(dim=-1)

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		if update_alpha:
			self.log_alpha_optimizer.zero_grad()
			alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

			if L is not None:
				L.log('train_alpha/loss', alpha_loss, step)
				L.log('train_alpha/value', self.alpha, step)

			alpha_loss.backward()
			self.log_alpha_optimizer.step()

	def soft_update_critic_target(self):
		utils.soft_update_params(
			self.critic.Q1, self.critic_target.Q1, self.critic_tau
		)
		utils.soft_update_params(
			self.critic.Q2, self.critic_target.Q2, self.critic_tau
		)
		utils.soft_update_params(
			self.critic.encoder, self.critic_target.encoder,
			self.encoder_tau
		)

	def update_warm_up(self, replay_buffer, L, step):


		obs, action, reward, next_obs, not_done = replay_buffer.sample_sac(n=128)

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()



	def update(self, replay_buffer, L, step,transform_strength):


		obs, action, reward, next_obs, not_done = replay_buffer.sample_sac(n=64)
		if (transform_strength>0):
			flatten_obs = obs_to_tensors(obs.clone(),32)
			image=self.pipe.image_processor.preprocess(flatten_obs).to(torch.float16)
			init_latents=self.pipe.vae.config.scaling_factor *retrieve_latents(self.pipe.vae.encode(image),generator=None)
			shape = init_latents.shape
			noise = randn_tensor(shape, generator=None, device="cuda", dtype=torch.float16)
			init_latents = self.pipe.scheduler.add_noise(init_latents, noise, torch.Tensor([transform_strength]).cuda().long())
			generated_images = self.pipe(prompt="", 
						num_inference_steps=1,
				guidance_scale=0.0,height=32,width=32,
				num_images_per_prompt=init_latents.shape[0],latents=init_latents,
				timesteps=transform_strength,
				output_type='pt').images 
			
			output_diffusion = torch.clamp(255 * generated_images,0,255)
			L.log('train/ES_STEP',transform_strength,step)
			output_diffusion=output_diffusion.to(torch.float32)
			aug_obs = F.interpolate(output_diffusion,size=(84,84),mode='bilinear', align_corners=False).view(-1,9,84,84).cuda()
			aug_obs = obs*self.args.mixing_ratio + aug_obs*(1-self.args.mixing_ratio)

			obs=torch.cat([obs,aug_obs],dim=0)
			action=torch.cat([action,action],dim=0)
			reward=torch.cat([reward,reward],dim=0)
			next_obs=torch.cat([next_obs,next_obs],dim=0)
			not_done=torch.cat([not_done,not_done],dim=0)


		self.update_critic(obs, action, reward, next_obs, not_done, L, step)
		


		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()


