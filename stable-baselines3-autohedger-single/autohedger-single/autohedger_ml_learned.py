import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import hedging_env

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from hedging_env import HedgingEnv
from dashboard import dashboard_axes
from stable_baselines3.common.noise import ActionNoise, NormalActionNoise

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def singleModelPredict(model_name, cycle=5, interactive=True):
    if interactive:
        plt.ion()

    env = HedgingEnv()
    #env.mu = -0.5
    model = SAC.load(model_name)
    obs = env.reset()
    cnt = 0

    while True:
        reward_history = []
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        reward_history += [reward]
        if done:
            env.render()
            obs = env.reset()
            cnt += 1
            if cnt > cycle:
                break


    if interactive:
        plt.ioff()

def multiModelPredict(cycle, interactive, *argv):
    if interactive:
        plt.ion()

    figure, axes = plt.subplots(len(argv), 4 , figsize=(12, 6))
    figure.tight_layout()

    menv = HedgingEnv()
    menv.reset()
    market = menv.market

    envs = []
    obsrv = []
    models = []
    actions = []

    for model_name in argv:
        env = HedgingEnv()
        env.model_name = model_name
        envs += [env]
        models += [SAC.load(model_name)]
        obsrv += [env.reset(market)]
        actions += [None]

    while True:
        done = False
        for i in range(len(models)):
            actions[i], _states = models[i].predict(obsrv[i])
            obsrv[i], _reward, done, _ = envs[i].step(actions[i])

        if done:
            menv = HedgingEnv()
            menv.reset()
            market = menv.market

            # plot
            for i in range(len(envs)):
                axs = axes[i]
                env = envs[i]
                dashboard_axes(env, axs[0], axs[1], axs[2], axs[3])

            plt.show()
            plt.pause(0.005)  # pause a bit so that plots are updated

            for i in range(len(envs)):
                obsrv[i] = envs[i].reset(market)


    if interactive:
        plt.ioff()

#singleModelPredict("./models/sac_autohedger_single_demo", 10, True)
multiModelPredict(0, True, "./models/sac_autohedger_single_demo", "./models/sac_autohedger_skew_demo")