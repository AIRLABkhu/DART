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
import augmentations
from algorithms.sac import SAC

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




class SVEA_C(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta
		self.args=args
	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if self.svea_alpha == self.svea_beta:
			obs = torch.cat([obs, augmentations.random_conv(obs.clone())], axis=0)
			action = torch.cat([action, action], axis=0)
			target_Q = torch.cat([target_Q, target_Q], axis=0)

			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.svea_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_aug = augmentations.random_conv(obs.clone())
			current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
			critic_loss += self.svea_beta * \
				(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
			
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update(self, replay_buffer, L, step, transform_strength):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_svea(n=64)

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
