#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: gym.py
"""

import gym
import random
import numpy as np
import imageio

class GymBase():
    def __init__(self):
        pass

    def init_record(self):
        self.need_record = True
        self.record_data = []

    def save_record(self,gif_path,duration=0.1):
        if hasattr(self,'need_record') and self.need_record and len(self.record_data) > 0:
            imageio.mimsave(gif_path, self.record_data,duration = duration)

class Pong_v0(GymBase):
    def __init__(self):
        super().__init__()

        env = gym.make('Pong-v0')
        env = env.unwrapped
        self.render = env.render
        self.close = env.close
        self.action_space = env.action_space
        self.action_size = 3
        self.action_map = {0: 2, 1: 3, 2: 0}

        self.observation_space = env.observation_space

        self.env = env
        self.pre_obs = None
        self.inner_done = False

    def reset(self):
        self.pre_obs = None

        if self.inner_done:
            return self.obs_dealer(self.env.reset())
        else:
            obs, reward, done, info = self.env.step(self.sample_action())
            return self.obs_dealer(obs)

    def obs_dealer(self, obs):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = obs[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        obs = I.astype(np.float).ravel()

        # print(np.shape(self.pre_obs))
        if self.pre_obs is None:
            return_obs = np.zeros(6400)
        else:
            return_obs = obs - self.pre_obs

        self.pre_obs = obs
        return return_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(self.action_map[action])

        if hasattr(self,'need_record') and self.need_record:
            self.record_data.append(obs)

        self.inner_done = done

        return self.obs_dealer(obs), reward, True if reward != 0 else False, info

    def sample_action(self):
        action = random.randint(0, self.action_size - 1)

        return action



env_id_map = {'Pong-v0': Pong_v0}


def make(env_name):
    if env_name in env_id_map:
        return env_id_map[env_name]()
    else:
        print('environment {} does\'t exist!'.format(env_name))
