"""
Created by Alexey Bakshaev (alex.bakshaev@gmail.com), 2020
"""
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import portfolio_hedging_env
from datetime import datetime
import os

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from portfolio_hedging_env import PortfolioHedgingEnv
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise

print(torch.__version__)

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def callback(locals, globals):
    reward = 0
    pos = locals['self'].replay_buffer.pos
    if pos > 0: reward = locals['self'].replay_buffer.rewards[pos-1]

    global reward_history
    reward_history += [reward]
    if env.timestamp % 10 == 0:
        plot_rewards()

def plot_rewards():
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(reward_history, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    if env.save_figs and env.timestamp % 500 == 0:
        plt.savefig("figs/progress_" + str(datetime.now().timestamp()) + "_ " + str(env.timestamp) + ".png")
    else:
        plt.pause(0.005)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())



env = PortfolioHedgingEnv(use_skew = False, hedger_verbose = False, corr = 0.0, instr_weight = 0.5, save_figs = True)
print("Inst weight: w = " + str(env.instr_weight))
env.model_name = "sac_autohedger_portfolio_common_c_0_w_0"

policy_args = {"net_arch" :  [8000, 8000]}

reward_history = []
noise = NormalActionNoise(0, 50)
model = SAC(MlpPolicy,
            env,
            verbose=2,
            learning_rate=2e-6,
            target_update_interval=256,
            learning_starts=5000,
            use_sde_at_warmup=True,
            use_sde=False,
            policy_kwargs=policy_args,
            buffer_size=int(10e6)
            )

model.learn(total_timesteps=20000, log_interval=50, n_eval_episodes=100, callback=callback)
#model.save(env.model_name)

plt.ioff()

del model
