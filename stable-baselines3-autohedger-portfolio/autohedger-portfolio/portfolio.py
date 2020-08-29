"""
Created by Alexey Bakshaev (alex.bakshaev@gmail.com), 2020
"""
from instrument_context import InstrumentContext, Price
import math
from hedger import Hedger

class Portfolio:
    def __init__(self, nsteps, corr, instr_weight, verbose):
        """
        simple portfolio of two correlated assets
        :param nsteps:
        :param corr:
        """
        self.nsteps = nsteps
        self.instruments = {}
        self.corr = corr
        self.instr_weight = instr_weight
        self.verbose = verbose

        # add instruments into portfolio
        self.add_instruments()

        # in correlation discovery mode we choose to track the blended incoming client flow as a single position
        # cumulative client position accrued historically
        self.reset_portfolio()

    def add_instruments(self):
        # simple portfolio of two correlated assets
        # corr determines how correlated the prices are
        # instr_weight determines how much of instr_1 and instr_2 are in the portfolio
        # self.instruments[0] = InstrumentContext(0, self.nsteps, 1)
        self.instruments[0] = InstrumentContext(0, self.nsteps, self.instr_weight)
        self.instruments[1] = InstrumentContext(1, self.nsteps, 1 - self.instr_weight,
                                                corr=self.corr)  # correlated instrument

    def get_num_instruments(self):
        return len(self.instruments)

    # def calc_weighted_price(self, p1, p2):
    #    # calculate weighted price of two instruments for the client portfolio
    #    return self.instr_weight * p1 + (1-self.instr_weight) * p2

    def add_client_trade(self, size, price: Price):
        """
            Changes position and pnl according to received CLIENT trade size
            :param size: CLIENT trade size, negative - client buys, positive - client sells
            :param tradePrice: trade execution price
            :param midPrice: reference market price
        """
        # if we buy (client sells) at bid < mid, we accrue positive pos and positive client pnl
        # if we sell (client buys) at ask > mid, we accrue negative pos and positive client pnl

        self.client_position += size

        if size > 0:
            trade_price = price.bid
        else:
            trade_price = price.ask

        client_spread = price.mid - trade_price
        self.client_pnl += size * client_spread
        self.client_half_spread += [abs(client_spread)]
        if self.verbose:
            print("Client: size: ", size, "mid", price.mid, " spread: ", client_spread,
                  " pnl_adj: ", size * client_spread)

    def revalue(self, mid_price):
        """
            Revalues existing position at a new market price
        """
        # e.g. 100 - >
        if self.last_price == 0:
            self.last_price = mid_price
            return

        market_move = mid_price - self.last_price
        self.client_market_pnl += self.client_position * market_move
        self.last_price = mid_price

        if self.verbose:
            print("Client market reval: ", "client pos: ", self.client_position, "mid", mid_price, " spread: ",
                  market_move,
                  " pnl_adj: ", self.client_position * market_move)

    def reset(self, market=None):
        self.reset_portfolio()

        # market data
        if market is None:
            # reset instr 0 market, extract random process
            if len(self.instruments) == 1:
                self.instruments[0].market.reset()
                self.mid_price_tensor = self.instruments[0].market.mid_price
            else:
                self.instruments[0].market.reset()
                master_process = self.instruments[0].market.brownian_driver
                assert master_process is not None
                # init instr 1 market with that process
                self.instruments[1].market.reset(master_process)
                # blend reference prices of two instruments for the client portfolio
                self.mid_price_tensor = self.instr_weight * self.instruments[0].market.mid_price + \
                                        (1 - self.instr_weight) * self.instruments[1].market.mid_price

        else:
            assert len(market) == len(self.instruments)
            self.mid_price_tensor = self.instr_weight * market[0].mid_price + \
                                    (1 - self.instr_weight) * market[1].mid_price
            for i in range(len(self.instruments)):
                self.instruments[i].market = market[i]

        # hedgers init
        for i in range(len(self.instruments)):
            self.instruments[i].hedger = Hedger(self.instruments[i].instrument_name, self.verbose)

    def reset_portfolio(self):
        self.client_pnl = 0.
        self.client_pnl_hist = []
        self.total_pnl_hist = []
        self.position_hist_client = [0.]
        self.position_hist_times = [0]
        self.client_half_spread = []
        self.client_position_hist = [0.]
        self.client_position = 0.
        self.position_value_hist_client = [0.]
        self.position_value_hist_net = [0.]
        self.last_price = 0.
        self.client_market_pnl = 0.
        self.client_market_pnl_hist = []
        self.mid_price_tensor = None

    def limit_penalty(self, max_pos_limit, is_linear=False):
        """
        For a given maximum position limit, calculate the overall limit penalty as Euclidean distance over assets
        :param max_pos_limit:
        :param is_linear:
        :return:
        """
        penalty = 0.0
        for instr in self.instruments.values():
            instr_penalty = 0.0

            if is_linear:
                instr_penalty = abs(instr.hedger.position) * instr.price_t0
            else:
                alpha = 3.0 * (math.exp(abs(instr.hedger.position) / max_pos_limit) - 1)
                instr_penalty = alpha * max_pos_limit * instr.price_t0

            penalty += instr_penalty ** 2

        return math.sqrt(penalty)

    def limit_penalty_soft(self, max_pos_limit, is_linear=False):
        """
        For a given maximum position limit, calculate the overall limit penalty as Euclidean distance over assets
        :param max_pos_limit:
        :param is_linear:
        :return:
        """
        eucl_pos = 0.0
        for instr in self.instruments.values():
            eucl_pos += (abs(instr.hedger.position)) ** 2

        eucl_pos = math.sqrt(eucl_pos)

        if is_linear:
            return eucl_pos * instr.price_t0
        else:
            alpha = 3.0 * (math.exp(eucl_pos / max_pos_limit) - 1)
            return alpha * max_pos_limit * instr.price_t0

    def getPortfolioValue(self, timestamp):
        net_value = 0.0
        for instr in self.instruments.values():
            instr_value = instr.hedger.position * instr.market.mid_price.numpy()[timestamp]
            if self.verbose:
                print(instr.instrument_name + " pos value: " + str(instr_value))
            net_value += instr_value

        portfolio_mid_price = self.mid_price_tensor.numpy()[timestamp]
        client_position_value = portfolio_mid_price * self.client_position
        if self.verbose:
            print("Client pos value: " + str(client_position_value))
        net_value += client_position_value

        return net_value

    def getMaxHedgeLeg(self, timestamp):
        instr1_val = self.instruments[0].hedger.position * self.instruments[0].market.mid_price.numpy()[timestamp]
        instr2_val = self.instruments[1].hedger.position * self.instruments[1].market.mid_price.numpy()[timestamp]
        client_position_value = self.client_position * self.mid_price_tensor.numpy()[timestamp]
        return max(instr1_val + client_position_value, instr2_val + client_position_value)

    def limit_penalty_portfolio(self, max_pos_limit, timestamp, is_linear=False):
        """
        For a given maximum net portfolio value limit, calculate the overall limit penalty as net portfolio value
        :param max_pos_limit:
        :param is_linear:
        :return:
        """

        net_value = abs(self.getPortfolioValue(timestamp))

        if is_linear:
            return net_value
        else:
            alpha = 3.0 * (math.exp(net_value / max_pos_limit / self.instruments[0].price_t0) - 1)
            return alpha * max_pos_limit  # * self.instruments[0].price_t0

    def relative_penalty_portfolio(self, max_pos_limit, timestamp, is_linear=False):
        """
        For a given maximum net portfolio value limit, calculate the overall limit penalty as net portfolio value
        :param max_pos_limit:
        :param is_linear:
        :return:
        """
        portfolio_mid_price = self.mid_price_tensor.numpy()[timestamp]
        client_position_value = portfolio_mid_price * self.client_position

        overhedge = .0
        total_hedge_value = .0

        for instr in self.instruments.values():
            hedge_value = instr.hedger.position * instr.market.mid_price.numpy()[timestamp]
            # print(instr.instrument_name + " pos value: " + str(hedge_value))
            total_hedge_value += hedge_value
            if (-hedge_value > client_position_value > 0) or \
                    (-hedge_value < client_position_value < 0):
                overhedge += abs(hedge_value + client_position_value)
                if self.verbose:
                    print(instr.instrument_name + ": balooning hedge: overhedge " + str(
                        abs(hedge_value + client_position_value)))

            elif self.client_position * instr.hedger.position > 0:
                overhedge += abs(hedge_value + client_position_value)
                if self.verbose:
                    print(instr.instrument_name + ": inverse hedge: overhedge " + str(
                        abs(hedge_value + client_position_value)))

        i1_hedge_value = self.instruments[0].hedger.position * self.instruments[0].market.mid_price.numpy()[timestamp]
        i2_hedge_value = self.instruments[1].hedger.position * self.instruments[1].market.mid_price.numpy()[timestamp]
        # don't want create synthetic (offsetting) positions with our hedgers
        # if i1_hedge_value * i2_hedge_value < 0:
        #    overhedge += abs(i1_hedge_value - i2_hedge_value)

        net_value = client_position_value + total_hedge_value

        if self.verbose:
            print("Client pos value: " + str(client_position_value) + " total_hedge_val: " + str(total_hedge_value)
                  + " overhedge: " + str(overhedge))

        penalty = abs(net_value) + overhedge * 5
        scaled_penalty = .0

        if is_linear:
            scaled_penalty = penalty
        else:
            alpha = 3.0 * (math.exp(penalty / max_pos_limit / self.instruments[0].price_t0) - 1)
            scaled_penalty = alpha * max_pos_limit

        # print("Penalty: " + str(scaled_penalty))
        return scaled_penalty

    def getPortfolioMarketStepPNL(self):
        return self.client_market_pnl_hist[-1]

    def getPortfolioClientStepPNL(self):
        return self.client_pnl_hist[-1]

    def getPortfolioMarketStepPNLdiff(self):
        if len(self.client_market_pnl_hist) == 0:
            return 0
        elif len(self.client_market_pnl_hist) == 1:
            return self.client_market_pnl_hist[-1]
        else:
            return self.client_market_pnl_hist[-1] - self.client_market_pnl_hist[-2]

    def getPortfolioClientStepPNLdiff(self):
        if len(self.client_pnl_hist) == 0:
            return 0
        elif len(self.client_pnl_hist) == 1:
            return self.client_pnl_hist[-1]
        else:
            return self.client_pnl_hist[-1] - self.client_pnl_hist[-2]

    def getPortfolioPositionsDiffState(self):
        # client trades
        positions = []
        client_pos_diff = 0
        if len(self.position_hist_client) == 1:
            client_pos_diff = self.position_hist_client[-1]
        else:
            client_pos_diff = self.position_hist_client[-1] - self.position_hist_client[-2]

        positions += [client_pos_diff]

        # hedge
        for instr in self.instruments.values():
            positions += [instr.hedger.getHedgePositionDiff()]

        return positions

    def portfolioStepPnl(self):
        """
        Calculate portfolio step PNL
        :return: portfolio PNL
        """
        pnl = 0
        for instr in self.instruments.values():
            pnl += instr.hedger.getStepPNL()

        pnl += self.getPortfolioMarketStepPNL()
        pnl += self.getPortfolioClientStepPNL()
        return pnl

    def portfolioStepPnlDiff(self):
        """
        Calculate portfolio step PNL
        :return: portfolio PNL
        """
        pnl = 0
        for instr in self.instruments.values():
            pnl += instr.hedger.getStepPNLdiff()

        pnl += self.getPortfolioMarketStepPNLdiff()
        pnl += self.getPortfolioClientStepPNLdiff()
        return pnl

    # def getMaxAbsPosition(self):
    #    maxPos = .0
    #    for instr in self.instruments.values():
    #        if abs(instr.hedger.position) > maxPos:
    #            maxPos = abs(instr.hedger.position)
    #    return maxPos

    def get_state(self, timestamp):
        net_value = 0.
        portfolio_mid_price = self.mid_price_tensor.numpy()[timestamp]
        client_portfolio_value = self.client_position * portfolio_mid_price
        net_value += client_portfolio_value

        state = []
        # state += [ self.client_position ]
        stateStr = "Hedger: "

        for instr in self.instruments.values():
            # hedger_prev = 0
            # if timestamp > 0:
            #    hedger_prev = instr.hedger.position * instr.market.mid_price.numpy()[timestamp-1]
            hedger_pos_value = instr.hedger.position * instr.market.mid_price.numpy()[timestamp]
            # hedger_diff = hedger_pos_value - hedger_prev

            # if abs(client_portfolio_value) < 100.0:
            #    state += [ hedger_pos_value / portfolio_mid_price ]
            # else:
            #    state += [ hedger_pos_value / client_portfolio_value]

            state += [hedger_pos_value / 1000.0]
            # state += [ ( hedger_pos_value + client_portfolio_value) / portfolio_mid_price]
            # state += [hedger_diff]

            net_value += hedger_pos_value
            stateStr += instr.instrument_name + " pos: " + str(instr.hedger.position)

        # stateStr += " Client pos: " + str(self.client_position)
        # state += [ self.client_position  ]

        # if abs(client_portfolio_value) < 100.0:
        state += [net_value / 1000.0]
        # else:
        #    state += [net_value / client_portfolio_value]


        # state = self.getPortfolioPositionsDiffState()

        if self.verbose:
            print("State: " + str(state) + " details: " + stateStr)

        return state

    def step(self, actions, timestamp):
        # actions: hedge via instrument set
        # assert len(actions) == len(self.instruments)
        if self.verbose:
            print("\nStep: " + str(timestamp))
            print("Actions: " + str(actions))

        # revalue client part of the net portfoio position
        portfolio_mid_price = self.mid_price_tensor.numpy()[timestamp]
        self.revalue(portfolio_mid_price)
        portfolio_net_flow = .0

        weightedPrice = Price(.0, .0, .0)

        # total_hedge_amount = actions[0]
        # w = actions[1]

        hedge_amounts = []
        # hedge_amounts += [ w * total_hedge_amount ]
        # hedge_amounts += [ (1 - w) * total_hedge_amount]

        for i in range(len(self.instruments)):
            # hedge_amount = hedge_amounts[i]
            hedge_amount = actions[i]
            # if self.use_skew:
            #    skew = action[1]

            # NOTE: hedgers here will only have hedge positions, not netted with the client one, since
            # the client position is separately tracked at portfolio level
            assert self.instruments[i].hedger.accrued_client_position == 0

            # revalue existing position at market rate
            mid_price = self.instruments[i].market.mid_price.numpy()[timestamp]

            # revalue hedge part of the net portfolio position
            self.instruments[i].hedger.revalue(mid_price)

            # hedge out the chosen size based on new market data
            self.instruments[i].hedger.addHedgeTrade(hedge_amount, mid_price,
                                                     self.instruments[i].market.hedge_price_bid.numpy()[timestamp],
                                                     self.instruments[i].market.hedge_price_ask.numpy()[timestamp])

            # accrue position change from Poisson client trade flow
            net_flow = self.instruments[i].market.flow_bid.numpy()[timestamp] - \
                       self.instruments[i].market.flow_ask.numpy()[timestamp]
            baseline_flow = self.instruments[i].market.baseline_flow.numpy()[timestamp]
            bid_price = self.instruments[i].market.client_price_bid.numpy()[timestamp]
            ask_price = self.instruments[i].market.client_price_ask.numpy()[timestamp]
            adj_client_flow = 0

            # self.hedger.addUnskewedClientTrade(net_flow, mid_price, bid_price, ask_price)

            # if self.use_skew:
            # bid_price, ask_price, adj_client_flow = \
            # mm.calc_skewed_price(skew, baseline_flow, mid_price, bid_price, ask_price, self.hedge_size_limit)
            # net_flow -= adj_client_flow

            weightedPrice += self.instruments[i].weight * Price(mid_price, bid_price, ask_price)
            portfolio_net_flow += self.instruments[i].weight * net_flow
            if self.verbose:
                print(
                    "Instr: " + self.instruments[i].instrument_name + " flow: " + str(net_flow) + " " + Price(mid_price,
                                                                                                              bid_price,
                                                                                                              ask_price).as_string())

            self.instruments[i].hedger.step()

        # if skew is used, this is both skewed size and price
        if self.verbose:
            print("Client: " + "net flow: " + str(portfolio_net_flow) + " " + weightedPrice.as_string())

        self.add_client_trade(portfolio_net_flow, weightedPrice)

        self.client_market_pnl_hist += [self.client_market_pnl]
        self.client_pnl_hist += [self.client_pnl]

        self.position_hist_times += [timestamp]
        self.position_hist_client += [self.client_position]

        pos_val_client = self.mid_price_tensor[timestamp] * self.client_position
        pos_val_net = pos_val_client + self.instruments[0].hedger.position_hist_hedge_value[timestamp] + \
                      self.instruments[1].hedger.position_hist_hedge_value[timestamp]
        self.position_value_hist_client += [pos_val_client]
        self.position_value_hist_net += [pos_val_net]
        self.total_pnl_hist += [self.portfolioStepPnl()]