"""
Created by Alexey Bakshaev (alex.bakshaev@gmail.com), 2020
"""
import matplotlib.pyplot as plt
from statistics import mean
import dashboard as db

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from hedging_env import HedgingEnv, RewardType
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise
import os

plt.ion()

def callback(locals, globals):
    env.hedger.verbose = False
    reward = 0
    pos = locals['self'].replay_buffer.pos
    if pos > 0: reward = locals['self'].replay_buffer.rewards[pos-1]

    global reward_history
    reward_history += [reward]
    if env.timestamp % 5 == 0:
        db.plot_rewards(env, reward_history)

env = HedgingEnv(use_skew=True, skew_beta=0.5, reward_type=RewardType.MaxPnl)
env.model_name = "sac_autohedger_skew"

policy_args = {"net_arch" :  [2048, 2048]}

reward_history = []
client_spread_history = []
hedge_spread_history = []
noise = NormalActionNoise(0, 50)
model = SAC(MlpPolicy, env,
            verbose=2,
            learning_rate=5e-6,
            target_update_interval=32,
            learning_starts=20,
            use_sde_at_warmup=True,
            use_sde=False,
            policy_kwargs=policy_args,
            )

model.learn(total_timesteps=7000, log_interval=50, n_eval_episodes=1000, callback=callback)

client_mean_spread = mean(env.hedger.client_half_spread)
hedge_mean_spread = mean(env.hedger.hedge_half_spread)
print("Client spread: " + str(client_mean_spread))
print("Hedge spread: "+ str(hedge_mean_spread))

#model.save(os.path.dirname(__file__) + "/models/" + env.model_name)

plt.ioff()

del model
