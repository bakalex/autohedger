"""
Created by Alexey Bakshaev (alex.bakshaev@gmail.com), 2020
"""
import statistics
from statistics import mean

class Hedger:
    """
    Class for tracking realized strategy PNL
    """
    def __init__(self, instr_name = None, verbose = False):
        self.instr_name = ''
        if instr_name is not None:
            self.instr_name = instr_name + ': '

        # net current position: client + hedge
        self.position = 0.
        # cumulative client position accrued historically
        self.accrued_client_position = 0.
        # cumulative hedge position
        self.accrued_hedge_position = 0.

        self.hedge_pnl = 0.
        self.client_pnl = 0.
        self.unskewed_client_pnl = 0.
        self.market_pnl = 0.
        self.last_price = 0.
        self.timestamp = 0

        self.pnl_hist = []
        self.client_pnl_hist = []
        self.unskewed_client_pnl_hist = []
        self.hedge_pnl_hist = []
        self.market_pnl_hist = []

        self.position_hist  = [0.]
        self.position_hist_hedge = [0.]
        self.position_hist_hedge_value = [0.]
        self.position_hist_client = [0.]
        self.position_hist_times = [0]

        self.hedge_trade_prices = []
        self.hedge_trade_sizes = []
        self.hedge_trade_times = []

        self.client_half_spread = []
        self.hedge_half_spread = []

        self.verbose = verbose

    def meanClientHalfSpread(self):
        return mean(self.client_half_spread)

    def meanHedgeHalfSpread(self):
        return mean(self.hedge_half_spread)

    def percStats(self):
        total = self.getPNL()
        return 100 * self.client_pnl / total, 100 * self.market_pnl / total, 100 * self.hedge_pnl / total

    def sharpeRatio(self):
        returns = []
        for i in range(2, len(self.pnl_hist)):
            returns += [self.pnl_hist[i] - self.pnl_hist[i-1]]

        return 100 * statistics.mean(returns) / statistics.stdev(returns)

    def revalue(self, mid_price):
        """
            Revalues existing position at a new market price
        """
        # e.g. 100 - >
        if self.last_price == 0:
            self.last_price = mid_price
            return

        market_move = mid_price - self.last_price
        self.market_pnl += self.position * market_move
        self.last_price = mid_price

        if self.verbose:
            print(self.instr_name + ": Market: Time: ", self.timestamp, " old pos: ", self.position, " mid", mid_price, " spread: ", market_move,
              " pnl_adj: ", self.position * market_move)

    def addClientTrade(self, size, mid_price, bid_price, ask_price):
        """
            Changes position and pnl according to received CLIENT trade size
            :param size: CLIENT trade size, negative - client buys, positive - client sells
            :param tradePrice: trade execution price
            :param midPrice: reference market price
        """
        # if we buy (client sells) at bid < mid, we accrue positive pos and positive client pnl
        # if we sell (client buys) at ask > mid, we accrue negative pos and positive client pnl
        self.position += size
        self.accrued_client_position += size

        if size > 0:
            trade_price = bid_price
        else:
            trade_price = ask_price

        client_spread = mid_price - trade_price
        self.client_pnl += size * client_spread
        self.client_half_spread += [abs(client_spread)]
        if self.verbose:
            print(self.instr_name + ": Client: Time: ", self.timestamp, " size: ", size, "mid", mid_price, " spread: ", client_spread,
                  " pnl_adj: ", size * client_spread)


    def addUnskewedClientTrade(self, size, mid_price, bid_price, ask_price):
        """
        Log unskewed client PNL. Doesn't impact actual net PNL.
            - if we buy (client sells) at bid < mid, we accrue positive pos and positive client pnl
            - if we sell (client buys) at ask > mid, we accrue negative pos and positive client pnl
        """
        if size > 0:
            trade_price = bid_price
        else:
            trade_price = ask_price

        client_spread = mid_price - trade_price
        self.unskewed_client_pnl += size * client_spread
        if self.verbose:
            print(self.instr_name + ": Unskewed client trade: Time: ", self.timestamp, " size: ", size, "mid", mid_price, " spread: ", client_spread,
                  " pnl_adj: ", size * client_spread)

    def addHedgeTrade(self, size, mid_price, bid_price, ask_price):
        """
            Changes position and pnl according to received HEDGE trade size
            :param size: HEDGE trade size, negative - we sell, positive - we buy
            :param tradePrice: trade execution price
            :param midPrice: reference market price
        """
        if size is 0:
            return

        # if we buy at ask > mid, we accrue positive pos and negative hedge pnl
        # if we sell at bid < mid, we accrue negative pos and negative hedge pnl
        self.position += size
        self.accrued_hedge_position += size

        if size > 0:
            trade_price = ask_price
        else:
            trade_price = bid_price

        hedge_spread = mid_price - trade_price
        self.hedge_pnl += size * hedge_spread
        self.hedge_half_spread += [abs(hedge_spread)]

        self.hedge_trade_prices += [trade_price]
        self.hedge_trade_sizes += [size]
        self.hedge_trade_times += [self.timestamp]

        if self.verbose:
            print(self.instr_name + ": Hedge: Time: ", self.timestamp, " size: ", size, "mid", mid_price, " spread: ", hedge_spread,
              " pnl_adj: ", size * hedge_spread)

    def getHedgePositionDiff(self):
        if len(self.position_hist_hedge) == 0:
            return 0
        elif len(self.position_hist_hedge) == 1:
            return self.position_hist_hedge[-1]
        else:
            return self.position_hist_hedge[-1] - self.position_hist_hedge[-2]

    def getPNL(self):
        return self.client_pnl + self.market_pnl + self.hedge_pnl

    def getStepPNL(self):
        return self.pnl_hist[-1]

    def getMarketStepPNL(self):
        return self.market_pnl_hist[-1]

    def getClientStepPNL(self):
        return self.client_pnl_hist[-1]

    def getHedgeStepPNL(self):
        return self.hedge_pnl_hist[-1]

    def getStepPNLdiff(self):
        if len(self.pnl_hist) == 0:
            return 0
        elif len(self.pnl_hist) == 1:
            return self.pnl_hist[-1]
        else:
            return self.pnl_hist[-1] - self.pnl_hist[-2]

    def getMarketStepPNLdiff(self):
        if len(self.market_pnl_hist) == 0:
            return 0
        elif len(self.market_pnl_hist) == 1:
            return self.market_pnl_hist[-1]
        else:
            return self.market_pnl_hist[-1] - self.market_pnl_hist[-2]

    def getClientStepPNLdiff(self):
        if len(self.client_pnl_hist) == 0:
            return 0
        elif len(self.client_pnl_hist) == 1:
            return self.client_pnl_hist[-1]
        else:
            return self.client_pnl_hist[-1] - self.client_pnl_hist[-2]

    def getHedgeStepPNLdiff(self):
        if len(self.hedge_pnl_hist) == 0:
            return 0
        elif len(self.hedge_pnl_hist) == 1:
            return self.hedge_pnl_hist[-1]
        else:
            return self.hedge_pnl_hist[-1] - self.hedge_pnl_hist[-2]

    def step(self):
        self.pnl_hist += [self.getPNL()]
        self.market_pnl_hist += [self.market_pnl]
        self.hedge_pnl_hist += [self.hedge_pnl]
        self.client_pnl_hist += [self.client_pnl]
        self.unskewed_client_pnl_hist += [self.unskewed_client_pnl]

        self.position_hist_times += [self.timestamp]
        self.position_hist += [self.position]
        self.position_hist_hedge += [self.accrued_hedge_position]
        self.position_hist_client += [self.accrued_client_position]
        self.position_hist_hedge_value += [self.accrued_hedge_position * self.last_price]

        self.client_half_spread = self.client_half_spread[-1000:]
        self.hedge_half_spread = self.hedge_half_spread[-1000:]

        self.timestamp += 1