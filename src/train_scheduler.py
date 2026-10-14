import sys
import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
import math
from PIL import Image
import random

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

total_episodes = 2000
# 1. Constance [50]
def constance_scheduler1(episode):
    return 1

def constance_scheduler50(episode):
    return 50

def constance_scheduler100(episode):
    return 100

def linear_up_scheduler(episode):
    return 1 + (99 / 400) * episode  # Scale from 1 to 100

def cosine_scheduler_start100(episode, T=100):
    return 100 - 99.5 * (1 - np.cos(2 * np.pi * episode / T)) / 2

def randomized_t_scheduler(args):

    return np.random.randint(1, args.tmax + 1)

def ERDAA(train_episode_rewards,step,T,args):
    if step < args.init_steps:
        return 0
    else:
        train_episode_rewards = np.array(train_episode_rewards)

        slopes = np.diff(train_episode_rewards)
        fluctuation_strength = np.std(slopes)
        if fluctuation_strength==0:
                return 1
        first = fluctuation_strength*args.k*args.tmax/1000
        second = fluctuation_strength*args.k
        residuals = np.diff(train_episode_rewards)
        residuals = first * np.tanh(residuals[-1] / second)
        T = np.clip(T+residuals, 1, args.tmax)
        return T



def evaluate(env, mode, agent, video, num_episodes, L, step, video_dir, test_env=False):
    episode_rewards = []



    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        images = []

        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward

            a = obs[6:9, :, :].copy()
            a = np.transpose(a, (1, 2, 0))  # (3, H, W) -> (H, W, 3)

            img = (a - a.min()) / (a.max() - a.min()) * 255
            img = img.astype(np.uint8)
            image = Image.fromarray(img)
            images.append(image)  


        if L is not None:
            _test_env = '_' + mode if test_env else ''
            gif_path = os.path.join(video_dir, f'{step}{_test_env}{i}.gif')
            if i%5==0:
                images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
            L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)

def main(args):
        # Set seed
        utils.set_seed_everywhere(args.seed)
        #torch.backends.cudnn.deterministic=True
        #torch.backends.cudnn.benchmark=False

        # Initialize environments
        gym.logger.set_level(40)
        env = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode='train'
        )


        test_env = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed+42,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode='color_easy',
                intensity=args.distracting_cs_intensity
        ) if args.eval_mode is not None else None
        test_env2 = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed+84,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode='color_hard',
                intensity=args.distracting_cs_intensity
        ) if args.eval_mode is not None else None


        test_env3 = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed+126,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode='video_easy',
                intensity=args.distracting_cs_intensity
        ) if args.eval_mode is not None else None

        test_env4 = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed+168,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                mode='video_hard',
                intensity=args.distracting_cs_intensity
        ) if args.eval_mode is not None else None    



        # Create working directory
        work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, str(args.seed)+'_'+str(args.tmax)+'_'+str(args.mixing_ratio))+'_'+str(args.img_size)+'_'+str(args.k)+args.er_scheduler
        args.tag=str(args.tmax)+'_'+str(args.mixing_ratio)+'_'+str(args.img_size)+'_'+str(args.k)+args.er_scheduler
        print(args.domain_name)
        print('Working directory:', work_dir)
        if os.path.exists(work_dir):
            delete_option = input('working dir already exists, delete it? y or n :')
            if 'y' == delete_option:
                import shutil
                shutil.rmtree(work_dir)
            else:
                assert os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
        if not os.path.exists(work_dir):
                utils.make_dir(work_dir)
        model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
        video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
        video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
        # utils.write_info(args, os.path.join(work_dir, 'info.log'))

        #rint(torch.cuda.is_available())
        # Prepare agent
        assert torch.cuda.is_available(), 'must have cuda enabled'
        replay_buffer = utils.ReplayBuffer(
                obs_shape=env.observation_space.shape,
                action_shape=env.action_space.shape,
                capacity=args.train_steps,
                batch_size=args.batch_size,
                args = args
        )
        cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
        print('Observations:', env.observation_space.shape)
        print('Cropped observations:', cropped_obs_shape)
        agent = make_agent(
                obs_shape=cropped_obs_shape,
                action_shape=env.action_space.shape,
                args=args
        )

        start_step, episode, episode_reward, done = 0, 0, 0, True
        L = Logger(work_dir)
        start_time = time.time()
        train_episode_rewards=[]
        transform_strength=1
        for step in range(start_step, args.train_steps+1):
                
                #a=time.time() 
                if done:
                        if step > start_step:
                                L.log('train/duration', time.time() - start_time, step)
                                start_time = time.time()
                                L.dump(step)

                        # Evaluate agent periodically
                        if step % args.eval_freq == 0 and step>0:
                                print('Evaluating:', work_dir)
                                L.log('eval/episode', episode, step)
                                mode='train'
                                evaluate(env,mode, agent, video, args.eval_episodes, L, step,video_dir=video_dir)
                                if test_env is not None:
                                        mode='color_easy'
                                        evaluate(test_env,mode, agent, video, args.eval_episodes, L, step,video_dir=video_dir, test_env=True)
                                        mode='color_hard'
                                        evaluate(test_env2,mode, agent, video, args.eval_episodes, L, step,video_dir=video_dir, test_env=True)
                                        mode='video_easy'
                                        evaluate(test_env3,mode, agent, video, args.eval_episodes, L, step,video_dir=video_dir, test_env=True)
                                        mode='video_hard'
                                        evaluate(test_env4,mode, agent, video, args.eval_episodes, L, step,video_dir=video_dir, test_env=True)
                                L.dump(step)

                        # Save agent periodically
                        if step > start_step and step % args.save_freq == 0:
                              torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

                        L.log('train/episode_reward', episode_reward, step)
                        train_episode_rewards.append(episode_reward)
                        obs = env.reset()


                        assert args.er_scheduler in ["constance_scheduler1","constance_scheduler50","constance_scheduler100","linear_up_scheduler","cosine_scheduler_start100","randomized_t_scheduler"]
                        if args.er_scheduler=="constance_scheduler1":
                               transform_strength=constance_scheduler1(episode)
                        elif args.er_scheduler=="constance_scheduler50":
                               transform_strength=constance_scheduler50(episode)
                        elif args.er_scheduler=="constance_scheduler100":
                               transform_strength=constance_scheduler100(episode)
                        elif args.er_scheduler=="linear_up_scheduler":
                               transform_strength=linear_up_scheduler(episode)
                        elif args.er_scheduler=="cosine_scheduler_start100":
                               transform_strength=cosine_scheduler_start100(episode)
                        elif args.er_scheduler=="randomized_t_scheduler":
                               transform_strength=randomized_t_scheduler(args)                    
                        
                        done = False
                        episode_reward = 0
                        episode_step = 0
                        episode += 1

                        L.log('train/episode', episode, step)

                # Sample action for data collection
                if step < args.init_steps:
                        action = env.action_space.sample()
                else:
                        with utils.eval_mode(agent):

                                action = agent.sample_action(obs)
                
                  
                # Run training update
                # Run training update
                if step >= args.init_steps:
                        num_updates = args.init_steps if step == args.init_steps else 1
                        for _ in range(num_updates):
                                agent.update(replay_buffer, L, step,transform_strength)

                # Take step
                next_obs, reward, done, _ = env.step(action)
                done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
                replay_buffer.add(obs, action, reward, next_obs, done_bool)
                episode_reward += reward
                obs = next_obs

                episode_step += 1
                
                #print(time.time()-a)
        print('Completed training for', work_dir)


if __name__ == '__main__':
        args = parse_args()
        main(args)