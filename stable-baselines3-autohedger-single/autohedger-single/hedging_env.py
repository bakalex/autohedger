"""
Created by Alexey Bakshaev (alex.bakshaev@gmail.com), 2020
"""
import gym
from gym import spaces
from gym.utils import seeding
import torch
import numpy as np
from os import path
import market_model as mm
import matplotlib
import matplotlib.pyplot as plt
import math
from enum import Enum
from datetime import datetime
import os
import dashboard as db
import hedger
from hedger import Hedger

class RewardType(Enum):
    MaxPnl = 1
    MeanRiskReward = 2

class TradeType(Enum):
    Client = 1
    Hedge = 2

class HedgingEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, use_skew = False, skew_beta = 0.5, reward_type = RewardType.MaxPnl, client_spread = 0.18, hedge_spread = 0.07):
        self.model_name = None
        self.timestamp = 0
        self.price_t0 = 100.0
        self.vol = 0.15
        self.mu = .0
        self.nsteps = 512
        self.client_spread = client_spread
        self.hedge_spread = hedge_spread
        self.hedge_size_limit = 32
        self.max_pos_limit = 256
        self.use_skew = use_skew
        self.skew_beta = skew_beta
        self.reward_type = reward_type
        print("Model: Skew: ", self.use_skew)
        self.skew_hist = []
        self.figure = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.ax4 = None

        self.market = mm.Market(self.nsteps, self.price_t0, self.vol, self.mu, self.client_spread, self.hedge_spread)
        self.viewer = None

        """
        Observation space:
        - current position (gives current risk appetite)
        Other candidates (not used):
            - price log-return (gives a measure of market PNL)
            - client ask premium (client_ask - hedge_ask)
            - client bid premium (hedge_bid - client_bid)
        """
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)

        if not self.use_skew:
            """
            Action space:
            - hedge amount
            """
            self.action_space = spaces.Box(
                low = -self.hedge_size_limit,
                high= +self.hedge_size_limit,
                shape=(1,),
                dtype=np.float32
            )
        else:
            """
            Action space:
            - hedge amount
            - skew amount
            """
            self.action_space = \
                spaces.Box(low=np.array([-self.hedge_size_limit, -1.0]),
                           high=np.array([ self.hedge_size_limit, 1.0]),
                           shape=(2,),
                           dtype=np.float32)


    #def seed(self, seed=None):

    def limit_penalty(self, is_linear = False):
        if is_linear:
            return abs(self.hedger.position) * self.price_t0
        else:
            alpha = 3.0 * (math.exp(abs(self.hedger.position) / self.max_pos_limit) - 1)
            return alpha *  self.max_pos_limit * self.price_t0
        # piecewise-linear:
            # penalty when nearing 50% of the position limit
            # limit_penalty =  0.5 * max(abs(self.hedger.position) - 0.1 * self.max_pos_limit, 0) * 100
            # + 1.5 * max(abs(self.hedger.position) - 0.5 * self.max_pos_limit, 0) * 100
            # + max(abs(self.hedger.position) - 0.8 * self.max_pos_limit, 0) * 100

    """
    Hedges out a given size (action)
    :param action: size to hedge
    """
    def step(self, action):
        hedge_amount = action[0]
        if self.use_skew:
            skew = action[1]
            self.skew_hist += [skew]
            #print("Skew: " + str(skew))

        # revalue existing position at market rate
        mid_price = self.market.mid_price.numpy()[self.timestamp]
        self.hedger.revalue(mid_price)
        # hedge out the chosen size based on new market data

        self.hedger.addHedgeTrade(hedge_amount, mid_price,
                                  self.market.hedge_price_bid.numpy()[self.timestamp],
                                  self.market.hedge_price_ask.numpy()[self.timestamp])

        # accrue position change from Poisson client trade flow
        net_flow = self.market.flow_bid.numpy()[self.timestamp] - self.market.flow_ask.numpy()[self.timestamp]
        baseline_flow = self.market.baseline_flow.numpy()[self.timestamp]
        bid_price = self.market.client_price_bid.numpy()[self.timestamp]
        ask_price = self.market.client_price_ask.numpy()[self.timestamp]
        adj_client_flow = 0

        #self.hedger.addUnskewedClientTrade(net_flow, mid_price, bid_price, ask_price)

        if self.use_skew:
            bid_price, ask_price, adj_client_flow = \
                mm.calc_skewed_price(skew, baseline_flow, mid_price, bid_price, ask_price, self.hedge_size_limit, self.skew_beta)
            net_flow -=  adj_client_flow

        # if skew is used, this is both skewed size and price
        self.hedger.addClientTrade(net_flow, mid_price, bid_price, ask_price)

        self.hedger.step()

        if self.reward_type == RewardType.MaxPnl:
            reward = self.hedger.getStepPNL() - self.limit_penalty(False)
        elif self.reward_type == RewardType.MeanRiskReward:
            reward = max(-abs(self.hedger.getClientStepPNL() + self.hedger.getHedgeStepPNL()),.0) + \
                     self.hedger.getMarketStepPNL() - self.limit_penalty(False)
        else:
            raise Exception("Unsupported reward type!")

        state = self.get_hedger_state()

        self.timestamp += 1

        done = False
        if self.timestamp == self.nsteps - 1:
            done = True

        if done:
            self.render()

        # print("PNL:", self.hedger.getPNL())

        # harsher penality for 1.5x limit violation
        if abs(self.hedger.position) > 1.5 * self.max_pos_limit:
            self.render()
            return np.array(state, dtype=np.float32), -5 * abs(self.hedger.getPNL()), True, {}

        return np.array(state, dtype=np.float32), reward, done, {}

    def get_hedger_state(self):
        """
        Observation space:
        - current position (gives current risk appetite)

        Not used at present:
        - price log-return (gives a measure of market PNL)
        - client ask premium (client_ask - hedge_ask)
        - client bid premium (hedge_bid - client_bid)
        """
        state = [
            self.hedger.position
            # self.log_returns[self.timestamp],
            # self.client_price_ask[self.timestamp] - self.hedge_price_ask[self.timestamp],
            # - self.client_price_bid[self.timestamp] + self.hedge_price_bid[self.timestamp],
            #self.market.rolling_vol.numpy()[self.timestamp]
        ]
        return state

    def reset(self, market = None):
        # market data
        if market is None:
            self.market.reset()
        else:
            self.market = market

        # hedger
        self.hedger = Hedger()
        self.timestamp = 0
        return self.get_hedger_state()

    def render(self, mode='human'):
        # plt.ion()
        if self.figure is None:
            self.figure, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 6))

        db.dashboard(self)
        #db.dashboard_price_only(self)

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

    return [-state[0]]

def heuristic_hedger(env, render=False):
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic_action(env, s)
        s, r, done, info = env.step(a)
        total_reward += r

        steps += 1
        if done: break

    #multiplot(env.mid_price,
    #          env.market.flow_bid - env.market.flow_ask,
    #          env.hedger.position_hist,
    #          env.hedger.pnl_hist,
    #          env.hedger.client_pnl_hist,
    #          env.hedger.market_pnl_hist)
    env.render()
    return total_reward


if __name__ == '__main__':
    env = HedgingEnv(use_skew = False)
    heuristic_hedger(env, render=True)

