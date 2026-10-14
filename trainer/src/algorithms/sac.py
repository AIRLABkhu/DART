import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
from augmentations import *
import io
import requests

def obs_to_tensors(obs):
    flatten_obs = obs.view(-1, 3, 84, 84)
    flatten_obs = F.interpolate(flatten_obs, size = (32,32), mode='bilinear', align_corners=False)
    return flatten_obs/255.0

def get_background_image_tensor(initial_latent, transform_strength, port):
    # serialize the (downsampled) observations for the generator
    buffer_initial = io.BytesIO()
    torch.save(initial_latent.cpu(), buffer_initial)
    buffer_initial.seek(0)

    # request URL and payload
    url = "http://localhost:{}/generate".format(port)
    files = {
        'initial_latent': ('initial_latent.pt', buffer_initial, 'application/x-pytorch')
    }
    data = {
        'transform_strength': transform_strength  # sent as form data
    }

    # send the request
    response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        response_buffer = io.BytesIO(response.content)
        generated_images = torch.load(response_buffer)
        return generated_images
    else:
        raise ValueError(f"Server returned an error: {response.status_code}, {response.text}")

class FeaturesHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


class SAC(object):
    def __init__(self, obs_shape, action_shape, args):
        self.discount = args.discount
        self.critic_tau = args.critic_tau
        self.encoder_tau = args.encoder_tau
        self.actor_update_freq = args.actor_update_freq
        self.critic_target_update_freq = args.critic_target_update_freq

        shared_cnn = m.SharedCNN(
            obs_shape, args.num_shared_layers, args.num_filters
        ).cuda()
        head_cnn = m.HeadCNN(
            shared_cnn.out_shape, args.num_head_layers, args.num_filters
        ).cuda()
        actor_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim),
        )
        critic_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim),
        )

        self.actor = m.Actor(
            actor_encoder,
            action_shape,
            args.hidden_dim,
            args.actor_log_std_min,
            args.actor_log_std_max,
        ).cuda()
        self.critic = m.Critic(critic_encoder, action_shape, args.hidden_dim).cuda()
        self.critic_target = deepcopy(self.critic)

        self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
        self.log_alpha.requires_grad = True
        self.target_entropy = -np.prod(action_shape)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999),weight_decay=args.critic_weight_decay,
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
        )
        self.args=args
        self.train()
        self.critic_target.train()
        self.hook = FeaturesHook(self.critic.encoder.head_cnn)

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
        _obs = torch.FloatTensor(_obs).cuda()
        _obs = _obs.unsqueeze(0)
        return _obs

    def select_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        if L is not None:
            L.log("train_critic/loss", critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
        _, pi, log_pi, log_std = self.actor(obs, detach=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if L is not None:
            L.log("train_actor/loss", actor_loss, step)
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(
                dim=-1
            )

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

            if L is not None:
                L.log("train_alpha/loss", alpha_loss, step)
                L.log("train_alpha/value", self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def soft_update_critic_target(self):
        utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
        utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
        utils.soft_update_params(
            self.critic.encoder, self.critic_target.encoder, self.encoder_tau
        )


    def update(self, replay_buffer, L, step,transform_strength):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_sac()

        aug_obs = obs.clone()
        if transform_strength>0:

            flatten_selected_obs = obs_to_tensors(aug_obs)  # downsample to 32x32, scale to [0, 1]
            aug_obs = get_background_image_tensor(flatten_selected_obs, transform_strength, port=self.args.port).to(torch.float32)
            aug_obs=255*F.interpolate(aug_obs,size=(84,84),mode='bilinear', align_corners=False).view(obs.shape[0],obs.shape[1],obs.shape[2],obs.shape[3]).to(obs.device)

            aug_obs =obs.clone()*self.args.mixing_ratio+ aug_obs*(1-self.args.mixing_ratio)  # blend: mixing_ratio * clean + (1 - mixing_ratio) * generated

        obs = torch.cat([obs, aug_obs], dim=0)
        action = torch.cat([action, action], dim=0)
        reward = torch.cat([reward, reward], dim=0)
        next_obs = torch.cat([next_obs, next_obs], dim=0)
        not_done = torch.cat([not_done, not_done], dim=0)





        self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
