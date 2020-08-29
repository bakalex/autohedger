"""
Created by Alexey Bakshaev (alex.bakshaev@gmail.com), 2020
"""
import gym
from gym import spaces
from portfolio import Portfolio
from gym.utils import seeding
import torch
import numpy as np
from os import path
import market_model as mm
import matplotlib
import matplotlib.pyplot as plt
import math
from datetime import datetime
from dashboard import dashboard

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class PortfolioHedgingEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, use_skew = False, hedger_verbose = False, corr = .0, instr_weight = 0.5, save_figs = False):
        #self.hedger
        self.model_name = None
        self.timestamp = 0
        self.nsteps = 512
        self.hedge_size_limit = 64
        self.max_pos_limit = 256
        self.use_skew = use_skew
        print("Model: Skew: ", self.use_skew)

        self.corr = corr
        self.instr_weight = instr_weight
        self.hedger_verbose = hedger_verbose
        self.portfolio = Portfolio(self.nsteps, self.corr, self.instr_weight, self.hedger_verbose)
        self.save_figs = save_figs

        self.viewer = None
        self.figure = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ax4 = None

        """
        Observation space:
        - current position (gives current risk appetite)
        - price log-return (gives a measure of market PNL)
        - client ask premium (client_ask - hedge_ask)
        - client bid premium (hedge_bid - client_bid)
        TODO: volatility rolling average?
        """

        # case 2: position values
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

        """
        Action space:
        - amount to hedge
        """
        self.action_space = spaces.Box(
                low = -self.hedge_size_limit,
                high= +self.hedge_size_limit,
                shape=(2,),
                dtype=np.float32
            )


    def seed(self, seed=None):
        print("No seed")


    """
    Hedges out a given size (action)
    :param action: size to hedge
    """
    #def step(self, hedge_amount, skew):
    def step(self, action):
        self.portfolio.step(action, self.timestamp)

        portfolio_step_pnl = self.portfolio.portfolioStepPnl()
        #portfolio_step_pnl = self.portfolio.portfolioStepPnlDiff()
        #portfolio_position_penality = self.portfolio.limit_penalty_portfolio(self.max_pos_limit, self.timestamp, is_linear=False)
        portfolio_position_penality = self.portfolio.relative_penalty_portfolio(self.max_pos_limit, self.timestamp, is_linear=False)
        reward =  portfolio_step_pnl - portfolio_position_penality
        state = self.portfolio.get_state(self.timestamp)
        portfolio_value = self.portfolio.getPortfolioValue(self.timestamp)
        if self.hedger_verbose:
            print("    Timestamp: " + str(self.timestamp))
            print("    Step PNL: " + str(portfolio_step_pnl))
            print("    Position penalty: " + str(-portfolio_position_penality))
            print("    Cumul PNL: " + str(self.portfolio.portfolioStepPnl()))

        self.timestamp += 1

        done = False
        if self.timestamp == self.nsteps - 1:
            done = True

        if done:
            self.render()
        # print("PNL:", self.hedger.getPNL())

        # harsher penality for huge limits violation
        #if abs(self.portfolio.getMaxHedgeLeg(self.timestamp)) > 0.5 * self.max_pos_limit * self.portfolio.instruments[0].price_t0:
        if abs(self.portfolio.getPortfolioValue(self.timestamp)) > 5 * self.max_pos_limit * self.portfolio.instruments[0].price_t0:
            self.render()
            return np.array(state, dtype=np.float32), -3 * abs(reward), True, {}

        if reward < -200000:
            self.render()
            return np.array(state, dtype=np.float32), -3 * abs(reward), True, {}

        return np.array(state, dtype=np.float32), reward, done, {}

    def reset(self, market = None):
        self.portfolio.reset(market)
        self.timestamp = 0
        return self.portfolio.get_state(self.timestamp)

    def render(self, mode='human'):
        # plt.ion()
        if self.figure is None:
            self.figure, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 6))

        dashboard(self)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

def heuristic_action(env, state):
    """
    Heuristic action (for testing of heuristic hedger):
        try to fully hedge out the current position (state[0])
        - latency: determines how frequently the position is hedged out

    returns:
         size to hedge out
    """
    latency = 1
    if env.timestamp % latency > 0:
        return 0

    new_hedge_amounts = []
    net_value = state[0]
    price = env.portfolio.instruments[0].market.mid_price.numpy()[env.timestamp]
    hedge_amount = - net_value / price
    for i in range(len(env.portfolio.instruments)):
        env.portfolio.instruments[i].market.mid_price.numpy()[env.timestamp]

    return [hedge_amount]

def heuristic_hedger(env, render=False):
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic_action(env, s)
        s, r, done, info = env.step(a)
        total_reward += r

        #if render:
            #still_open = env.render()
            #if still_open == False: break

        #if steps % 20 == 0 or done:
        #    print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
        #    print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break

    #multiplot(env.mid_price,
    #          env.market.flow_bid - env.market.flow_ask,
    #          env.hedger.position_hist,
    #          env.hedger.pnl_hist,
    #          env.hedger.client_pnl_hist,
    #          env.hedger.market_pnl_hist)
    #dashboard(env)
    env.render()
    return total_reward


if __name__ == '__main__':
    env = PortfolioHedgingEnv(use_skew = False, corr=1.0, instr_weight = 1)
    heuristic_hedger(env, render=True)

