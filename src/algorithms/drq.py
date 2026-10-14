import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m
from algorithms.sac import SAC

from .rl_utils import *
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt


def obs_to_tensors(obs):
    flatten_obs = obs.view(-1, 3, 84, 84)
    flatten_obs = F.interpolate(flatten_obs, size = (32,32), mode='bilinear', align_corners=False)
    return flatten_obs/255.0

class DrQ(SAC): # [K=1, M=1]
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.args=args
	def update(self, replay_buffer, L, step, transform_strength):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_drq()





		if (transform_strength>0):
			flatten_obs = obs_to_tensors(obs)
			self.pipeline.scheduler.config['ER_ratio']=transform_strength
			self.pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(self.pipeline.scheduler.config)
			output_diffusion = self.pipeline(prompt = [""]*flatten_obs.shape[0], image=flatten_obs, guidance_scale=0.0, num_inference_steps=1, strength=1,output_type='pt').images.to(self.device)
			L.log('train/ES_STEP',transform_strength,step)
			output_diffusion=output_diffusion.to(torch.float32)
			aug_obs = 255*F.interpolate(output_diffusion,size=(84,84),mode='bilinear', align_corners=False).view(-1,9,84,84)
			


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
