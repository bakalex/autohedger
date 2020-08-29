"""
Created by Alexey Bakshaev (alex.bakshaev@gmail.com), 2020
"""
import matplotlib
import matplotlib.pyplot as plt
import torch
from datetime import datetime

def tostr(value, ndigits = 0):
    if ndigits is 0:
        return str(round(value)).rstrip('0').rstrip('.')
    else:
        return str(round(value, ndigits))

def multiplot(*tensors):
    f_multiplot(False, *tensors)

def plot(*tensors):
    f_multiplot(True, *tensors)

def dashboard_single(env):
    #plt.ion()
    plt.clf()
    # fig = plt.figure(figsize=(10, 6))
    # price process
    #f, ax = plt.subplots(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.title('Price process')
    plt.plot(env.market.mid_price.numpy(), label="mid")
    plt.plot(env.market.client_price_ask.numpy(), label="client_ask")
    plt.plot(env.market.client_price_bid.numpy(), label="client_bid")
    plt.plot(env.market.hedge_price_ask.numpy(), label="hedge_ask")
    plt.plot(env.market.hedge_price_bid.numpy(), label="hedge_bid")
    plt.legend(prop={'size': 6})
    # PNL
    plt.subplot(2, 2, 2)
    plt.title('PNL')
    plt.plot(env.hedger.pnl_hist, label="Net")
    plt.plot(env.hedger.client_pnl_hist, label="Client")
    plt.plot(env.hedger.market_pnl_hist, label="Market")
    plt.plot(env.hedger.hedge_pnl_hist, label="Hedge")
    plt.legend(prop={'size': 6})
    # Position
    plt.subplot(2, 2, 3)
    plt.title('Position')
    plt.plot(env.hedger.position_hist, label="Net")
    plt.plot(env.hedger.position_hist_client, label="Client")
    plt.plot(env.hedger.position_hist_hedge, label="Hedge")
    plt.legend(prop={'size': 6})


    plt.show()
    plt.pause(0.005)  # pause a bit so that plots are updated
    #plt.ioff()

def dashboard(env):
    dashboard_axes_portfolio(env, env.ax1, env.ax2, env.ax3, env.ax4)

    if not env.save_figs:
        plt.show()
        plt.pause(0.005)  # pause a bit so that plots are updated
    else:
        plt.figure(env.figure.number)
        plt.savefig("figs/dashbrd_" + str(datetime.now().timestamp()) + "_ " + str(env.timestamp) + ".png", dpi=300)

def dashboard_axes(env, ax1, ax2, ax3, ax4):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # Price process
    ax1.set_title('Price process')
    ax1.plot(env.market.mid_price.numpy(), label="mid")
    ax1.plot(env.market.client_price_ask.numpy(), label="client_ask")
    ax1.plot(env.market.client_price_bid.numpy(), label="client_bid")
    ax1.plot(env.market.hedge_price_ask.numpy(), label="hedge_ask")
    ax1.plot(env.market.hedge_price_bid.numpy(), label="hedge_bid")
    ax1.legend(prop={'size': 6})
    # PNL
    ax2.set_title('PNL')
    ax2.plot(env.hedger.pnl_hist, label="Net")
    ax2.plot(env.hedger.client_pnl_hist, label="Client")
    ax2.plot(env.hedger.market_pnl_hist, label="Market")
    ax2.plot(env.hedger.hedge_pnl_hist, label="Hedge")
    ax2.legend(prop={'size': 6})
    # Position
    ax3.set_title('Position')
    ax3.plot(env.hedger.position_hist, label="Net")
    ax3.plot(env.hedger.position_hist_client, label="Client")
    ax3.plot(env.hedger.position_hist_hedge, label="Hedge")
    ax3.legend(prop={'size': 6})
    # Statistics
    ax4.set_title('Statistics')
    if env.model_name is not None:
        ax4.text(0.05, .9, "Model: " + env.model_name, fontsize=8)

    ax4.text(0.05, .8, "PNL: " + tostr(env.hedger.getPNL()), fontsize=8)

    cl, mkt, hdg = env.hedger.percStats()
    ax4.text(0.05, .7, "Client, Market, Hedge: " + tostr(cl, 2) + "% / " + tostr(mkt, 2) + "% / " + tostr(hdg, 2) + "%",
             fontsize=8)
    sharpe = env.hedger.sharpeRatio()
    ax4.text(0.05, .6, "Sharpe ratio: " + tostr(sharpe, 2) + "%", fontsize=8)

    ax4.set_yticklabels([])
    ax4.set_xticklabels([])


def dashboard_axes_portfolio(env, ax1, ax2, ax3, ax4):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # Price process
    ax1.set_title('Price processes')
    ax1.plot(env.portfolio.instruments[0].market.mid_price.numpy(), label="i1: mid")
    ax1.plot(env.portfolio.instruments[0].market.client_price_ask.numpy(), label="i1: client_ask")
    ax1.plot(env.portfolio.instruments[0].market.client_price_bid.numpy(), label="i1: client_bid")
    ax1.plot(env.portfolio.instruments[0].market.hedge_price_ask.numpy(), label="i1:hedge_ask")
    ax1.plot(env.portfolio.instruments[0].market.hedge_price_bid.numpy(), label="i1:hedge_bid")
    ax1.plot(env.portfolio.instruments[1].market.mid_price.numpy(), label="i2: mid")
    ax1.plot(env.portfolio.instruments[1].market.client_price_ask.numpy(), label="i2: client_ask")
    ax1.plot(env.portfolio.instruments[1].market.client_price_bid.numpy(), label="i2: client_bid")
    ax1.plot(env.portfolio.instruments[1].market.hedge_price_ask.numpy(), label="i2: hedge_ask")
    ax1.plot(env.portfolio.instruments[1].market.hedge_price_bid.numpy(), label="i2: hedge_bid")
    ax1.legend(prop={'size': 6})

    # Client portfolio value vs hedge positions value
    ax2.set_title('Portfolio vs Hedge values')
    ax2.plot(env.portfolio.position_value_hist_client, label="Client pos value")
    ax2.plot(env.portfolio.position_value_hist_net, label="Net")

    ax2.plot(env.portfolio.instruments[0].hedger.position_hist_hedge_value, label="i1: Hedge")
    ax2.plot(env.portfolio.instruments[1].hedger.position_hist_hedge_value, label="i2: Hedge")
    ax2.legend(prop={'size': 6})

    ax3.set_title('PNL')
    ax3.plot(env.portfolio.total_pnl_hist, label="Net")
    ax3.legend(prop={'size': 6})


def f_multiplot(same_plot, *tensors):
    n = len(tensors) + len(tensors) % 2

    plt.clf()

    i = 1
    for t in tensors:
        if not same_plot:
            plt.subplot(2, n/2, i)

        if torch.is_tensor(t):
            plt.plot(t.numpy())
        else:
            plt.plot(t)
            plt.title('Plot2')

        plt.title(i-1)
        i += 1

    plt.show()