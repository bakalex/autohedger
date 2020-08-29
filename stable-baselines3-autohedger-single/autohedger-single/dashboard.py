"""
Created by Alexey Bakshaev (alex.bakshaev@gmail.com), 2020
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from datetime import datetime

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

def tostr(value, ndigits = 0):
    if ndigits is 0:
        return str(round(value)).rstrip('0').rstrip('.')
    else:
        return str(round(value, ndigits))

def plot_rewards(env, reward_history, save_fig = True):
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(reward_history, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())

    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    if save_fig and env.timestamp % 500 == 0:
        plt.savefig(os.path.dirname(__file__) + "/figs/progress_" + str(datetime.now().timestamp()) + "_ " + str(env.timestamp) + ".png")

    plt.pause(0.005)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def plot_client_spreads(env, client_spread_history, hedge_spread_history, window = 2000):
    plt.figure(3)
    plt.clf()
    spreads_client = torch.tensor(client_spread_history, dtype=torch.float)
    spreads_hedge = torch.tensor(hedge_spread_history, dtype=torch.float)
    plt.xlabel('Episode')
    plt.ylabel('Average spreads')

    if len(spreads_client) >= window:
        means_c = spreads_client.unfold(0, window, 1).mean(1).view(-1)
        means_c = torch.cat((torch.zeros(window - 1), means_c)).numpy()
        plt.plot(means_c, label="Rolling avg mean client (m. maker) spreads")

    if len(spreads_hedge) >= window:
        means_h = spreads_hedge.unfold(0, window, 1).mean(1).view(-1)
        means_h = torch.cat((torch.zeros(window - 1), means_h)).numpy()
        plt.plot(means_h, label="Rolling avg mean hedge (m. taker) spreads")

    if len(spreads_client) >= window and len(spreads_hedge) >= window:
        plt.legend(prop={'size': 6})
    else:
        plt.text(0.1, 0.1, "Waiting to collect first " + str(window) + " data steps")

    if env.timestamp % 500 == 0:
        plt.savefig(os.path.dirname(__file__) + "/figs/client_spread_" + str(datetime.now().timestamp()) + "_ " + str(env.timestamp) + ".png")

    plt.pause(0.005)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def multiplot(*tensors):
    f_multiplot(False, *tensors)

def plot(*tensors):
    f_multiplot(True, *tensors)

def dashboard_price_only(env):
    #plt.ion()
    plt.clf()
    plt.title('Price process')
    plt.plot(env.market.mid_price.numpy(), label="mid")
    plt.plot(env.market.client_price_ask.numpy(), label="client_ask")
    plt.plot(env.market.client_price_bid.numpy(), label="client_bid")
    plt.legend(prop={'size': 6})

    plt.show()
    plt.pause(0.005)  # pause a bit so that plots are updated
    #plt.ioff()

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
    dashboard_axes(env, env.ax1, env.ax2, env.ax3, env.ax4)
    plt.figure(env.figure.number)
    plt.savefig(os.path.dirname(__file__) + "/figs/dashbrd_" + str(datetime.now().timestamp()) + "_ " + str(env.timestamp) + ".png")
    plt.show()
    plt.pause(0.02)  # pause a bit so that plots are updated

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

    ax4.set_yticklabels([])
    ax4.set_xticklabels([])

    txt_offset = 0
    if env.model_name is not None:
        ax4.text(0.05, .9 + txt_offset, "Model: " + env.model_name, fontsize=8)

    ax4.text(0.05, .8 + txt_offset, "PNL: " + tostr(env.hedger.getPNL()), fontsize=8)

    cl, mkt, hdg = env.hedger.percStats()
    ax4.text(0.05, .7 + txt_offset, "Client, Market, Hedge: " + tostr(cl, 2) + "% / " + tostr(mkt, 2) + "% / " + tostr(hdg, 2) + "%",
             fontsize=8)
    sharpe = env.hedger.sharpeRatio()
    ax4.text(0.05, .6 + txt_offset, "Sharpe ratio: " + tostr(sharpe, 2) + "%", fontsize=8)

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