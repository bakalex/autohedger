"""
Created by Alexey Bakshaev (alex.bakshaev@gmail.com), 2020
"""
import torch
from torch.distributions import Normal, LogNormal, Poisson
import numpy as np
import pandas as pd
import math
from numbers import Number
from enum import Enum
import statistics
from statistics import mean

def rolling_stdev(tensor, window=20, warmup=5):
    data = pd.DataFrame(data=tensor.numpy().flatten(), columns=['Values'])
    vol_pd = data['Values'].rolling(window, min_periods=1).std() * math.sqrt(len(tensor.numpy()))
    vol_np = vol_pd.fillna(vol_pd.head(warmup).mean()).values
    return torch.tensor(vol_np)

def rolling_mean(tensor, window=20, warmup=5):
    data = pd.DataFrame(data=tensor.numpy().flatten(), columns=['Values'])
    vol_pd = data['Values'].rolling(window, min_periods=1).mean()
    vol_np = vol_pd.fillna(vol_pd.head(warmup).mean()).values
    return torch.tensor(vol_np)

def rolling_sum(tensor, window=20, warmup=5):
    data = pd.DataFrame(data=tensor.numpy().flatten(), columns=['Values'])
    vol_pd = data['Values'].rolling(window, min_periods=1).sum()
    vol_np = vol_pd.fillna(vol_pd.head(warmup).mean()).values
    return torch.tensor(vol_np)

def gen_correlated(tensor, corr):
    """
        Returns a correlated normally distributed tensor.
        Assumes that input tensor is N(0,1)
    """
    eps = Normal(0, 1).sample((len(tensor),))
    return corr * tensor + math.sqrt(1-corr**2) * eps

def gen_price_process(nsteps=1000, x0=100, vol=0.15, mu=0.0, master_process=None, corr=0.0):
    """
        Simulates lognormal price process:
         - x0: starting value for the log-normal process
         - vol: volatility
         - mu=0: assumes martingale
         - master_process defines the master process this process is to be correlated with
         - corr: correlation to be used with brownian_master process to get the resulting log-normal process
        :return: price process, log_returns process, resulting Brownian driver

    """
    if master_process is not None:
        eps = gen_correlated(master_process, corr)
    else:
        eps = Normal(0, 1).sample((nsteps,))

    times = 1 / nsteps * torch.ones([nsteps, ])
    sqtimes = torch.sqrt(times)
    log_returns = mu * times - 0.5 * vol ** 2 * times + vol * eps * sqtimes
    log_x = math.log(x0) + torch.cumsum(log_returns, dim=0)
    process = torch.exp(log_x)
    return process, log_returns, eps

def gen_bid_ask(x0, x, vol=0.15, spread_mult = 0.8, stoch_spread_mult = 0.5, clampSpread = True, vol_window = 10):
    """
        Generates bid/ask based on the rolling vol of the input mid process and stochastic spread add-on
        Stochastic spread is a function of mid process volatility

        :param vol: volatility of mid process
        :param spread_mult: multiplier of resulting spread
        :param stoch_spread_mult: multiplier of sochastic addition on the top of rolling vol multiple
        :param clampSpread: determines whether to imply bounds on the resulting spread after stochastic lognormal addition is applied
    """
    nsteps=len(x)
    rolling_vol = rolling_stdev(x, vol_window) / math.sqrt(nsteps)

    spread_bid = rolling_vol + LogNormal(0, stoch_spread_mult * x0 * vol / math.sqrt(nsteps)).sample((nsteps,))
    spread_ask = rolling_vol + LogNormal(0, stoch_spread_mult * x0 * vol / math.sqrt(nsteps)).sample((nsteps,))

    # clamp resulting spread to guard against unreasonable samples from lognormal distr
    if(clampSpread):
        mean_vol = torch.mean(rolling_vol)
        torch.clamp(spread_bid, 0.1 * mean_vol, 2.5 * mean_vol)
        torch.clamp(spread_ask, 0.1 * mean_vol, 2.5 * mean_vol)

    x_bid = x - spread_mult * spread_bid
    x_ask = x + spread_mult * spread_ask
    return x_ask, x_bid

def normalize(tensor):
    return (tensor - torch.mean(tensor))/torch.std(tensor)

def gen_client_trade_flows(x, log_returns, window = 10, price_corr = 1, flow_returns_sensitivity = 10, alpha = 0.0):
    """
        Simulates bid and ask client trade sizes based on input price process x and its log_returns.
        Both x and log_returns are tensors representing simulation in time.
        Naively assumes that client net flow is correlated with the price process and overall client flow is a function of rolling volatility.

        :param price_corr: correlation between net client flow and the price process
        :param flow_returns_sensitivity: sensitivity of net client trade flow towards rolling mean log-returns
        :param alpha: allows to define the vol-independent "mean" client trade flow

        :return: Tensor of ask sizes,  Tensor of bid sizes, Tensor of baseline client order rate
    """
    rolling_returns = log_returns

    stdev_window = window
    if window > 5:  # allow for warmup
        rolling_returns = rolling_sum(log_returns, window)
    else:
        stdev_window = 10

    rolling_vol = rolling_stdev(x, stdev_window)

    # poisson rate: function of vol  + stoch adjustment
    flow_scaling_factor = 100
    beta = 1  # intensity scaling factor
      # constatnt daily flow
    client_trade_rate = flow_scaling_factor * (alpha + rolling_vol / torch.mean(rolling_vol))

    # net_intensity determines the direction of net client trade flow
    # assume that net client trade flow is correlated with rolling market returns
    # then net_intensity needs to be rescaled by flow_returns_sensitivity and also clipped

    # window = 1 for raw returns
    # net_intensity = flow_returns_sensitivity * rolling_returns
    net_intensity = flow_returns_sensitivity * gen_correlated(rolling_returns, price_corr)

    # return normalization - kills variance and somewhat decorrelates the flow from returns
    #net_intensity = flow_returns_sensitivity * gen_correlated(normalize(rolling_returns), price_corr)

    zero = torch.zeros([len(log_returns), ], dtype=torch.double)
    one = torch.ones([len(log_returns), ], dtype=torch.double)

    # net_intensity > 0: net outflow, client buys from us at our ask, we sell and unwind client pos
    # net_intensity < 0: net inflow, client sells to us at our bid, we buy and accrue client pos
    lmbda_floored_ask = torch.max(one + net_intensity, zero)
    lmbda_floored_bid = torch.max(one - net_intensity, zero)
    #print(lmbda_floored_ask, lmbda_floored_bid)
    #print(net_intensity)

    flow_ask = torch.poisson(client_trade_rate * lmbda_floored_ask)
    flow_bid = torch.poisson(client_trade_rate * lmbda_floored_bid)
    #print(flow_ask - flow_bid)
    return flow_ask, flow_bid, client_trade_rate

def calc_skewed_price(skew, client_trade_rate, mid, bid, ask, max_hedge_size, beta = 0.5):
    """
        Skew definition:
            s > 0: skewed_ask = ask - skew > mid. We sell cheaper and attract more client flow on ask side
            s < 0: skewed_bid = bid - skew < mid. We buy more expensively and attract more client flow on bid side

        This brackets skew in [bid - mid; ask - mid]
        Skew is defined as [-1, 1] box within the env, so need to rescale according to those expected buckets

        :param skew: price adjustment within [bid - mid; ask - mid]
        :param client_trade_rate: mean trade rate
        :param mid, bid, ask
        :param max_hedge_size: hedge env hedge increment limit
        :param beta: sensitivity of skewed trade flow to skew

        :return: skewed (or original) bid, skewed (or original) ask, client flow adjustment
    """
    adj_bid = bid
    adj_ask = ask
    adj_client_flow = 0

    if skew > 0:
        adj_ask = ask - skew * (ask - mid)  # rescaled from [0; 1]

    else:
        adj_bid = bid - skew * (mid - bid) # rescaled from [-1; 0]

#    if skew < bid - mid or skew > ask - mid:
#        raise Exception("Skewed price must be bracketed in [bid - mid; ask - mid]!")

    adj_client_flow = beta * max_hedge_size / client_trade_rate * skew
    return adj_bid, adj_ask, adj_client_flow

class Market:
    """
    Market data store & generator for hedging environment
    :param mid_corr: defines correlation to the master brownian process
    """
    def __init__(self, nsteps, price_t0, vol, mu, client_spread, hedge_spread, mid_corr = 0.0):
        self.nsteps = nsteps
        self.price_t0 = price_t0
        self.vol = vol
        self.mu = mu
        self.client_spread = client_spread
        self.hedge_spread = hedge_spread
        self.mid_corr = mid_corr

    def reset(self, master_process = None):
        """
        Resets the market accordning to Brownian price process and stochastic spreads
        :param master_process (optional) process to correlate the generated market with
        :return:
        """
        self.mid_price, self.log_returns, self.brownian_driver = gen_price_process(self.nsteps, self.price_t0, self.vol, self.mu,
                                                                            master_process, self.mid_corr)
        self.client_price_ask, self.client_price_bid = gen_bid_ask(self.price_t0, self.mid_price, self.vol,
                                                                      self.client_spread)
        self.hedge_price_ask, self.hedge_price_bid = gen_bid_ask(self.price_t0, self.mid_price, self.vol,
                                                                    self.hedge_spread)

        self.rolling_vol = rolling_stdev(self.mid_price, int(self.nsteps / 10))
        self.flow_ask, self.flow_bid, self.baseline_flow = gen_client_trade_flows(self.mid_price, self.log_returns,
                                                                                     window=5)

