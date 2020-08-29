"""
Created by Alexey Bakshaev (alex.bakshaev@gmail.com), 2020
"""
from hedger import Hedger
from market_model import Market

class Price:
"""
Helper class to facilitate blending of asset prices
"""
    def __init__(self, mid, bid, ask):
        self.mid = mid
        self.bid = bid
        self.ask = ask

    def __add__(self, other):
        return Price(self.mid + other.mid, self.bid + other.bid, self.ask + other.ask)

    def __mul__(self, w):
        return Price(w * self.mid, w * self.bid, w * self.ask)

    __rmul__ = __mul__

    def as_string(self):
        return "Price: m: " + str(self.mid) + " b: " + str(self.bid) + " a: " + str(self.ask)

class InstrumentContext:
    def __init__(self, ix, nsteps, weight, vol=0.15, mu=0.0, corr=0.0):
        self.instrument_name = 'instr_' + str(ix)
        self.corr = corr
        self.price_t0 = 100.0
        self.vol = vol
        self.mu = mu
        self.client_spread = 0.18
        self.hedge_spread = 0.07
        self.nsteps = nsteps
        self.market = Market(self.nsteps, self.price_t0, self.vol, self.mu, self.client_spread, self.hedge_spread, self.corr)
        self.weight = weight

    def reset(self, market=None, master_process=None):
        # market data
        if market is None:
            self.market.reset(master_process)
        else:
            self.market = market

        # plot(self.market.client_price_ask, self.market.client_price_bid, self.market.hedge_price_ask, self.market.hedge_price_bid)
        self.hedger = Hedger(self.instrument_name)