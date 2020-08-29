"""
Created by Alexey Bakshaev (alex.bakshaev@gmail.com), 2020
"""
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import dashboard as db

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy

plt.ion()

#env = gym.make('Pendulum-v0').env
env = gym.make('LunarLanderContinuous-v2').env
#env = gym.make('CartPole-v1')

def callback(locals, globals):
    global reward_history
    reward = 0
    pos = locals['self'].replay_buffer.pos
    if pos > 0: reward = locals['self'].replay_buffer.rewards[pos - 1]

    reward_history += [reward]
    db.plot_rewards(env, reward_history, False)
    env.render()

reward_history = []
policy_args = {"net_arch" :  [2048, 2048]}
model = SAC(MlpPolicy, env, verbose=2, policy_kwargs=policy_args)
model.learn(total_timesteps=10000, log_interval=1, n_eval_episodes=1000,callback=callback)



