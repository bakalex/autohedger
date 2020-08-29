# Autohedging / market-making with reinforcement learning
This code repository supports the paper: https://arxiv.org/pdf/2008.12275.pdf
The project makes use of an Open AI pytorch stable-baseline3 as a reinforcement-learning framework (soft actor-critic (SAC) implementation). The actual hedger code lives under directories autohedger-single (for the single asset version) and autohedger-portfolio (for the portfolio version). To simplify the installation, a version of Open AI baseline is supplied alongside with it, so that the code is runnable straight-away. 

## Paper abstract
The paper explores the application of a continuous action space soft actor-critic (SAC) reinforcement learning model to the area of automated market-making. The reinforcement learning agent receives a simulated flow of client trades, thus accruing a position in an asset, and learns to offset this risk by either hedging at simulated "exchange" spreads or by attracting an offsetting client flow by changing offered client spreads (skewing the offered prices). The question of learning minimum spreads that compensate for the risk of taking the position is being investigated. Finally, the agent is posed with a problem of learning to hedge a blended client trade flow resulting from independent price processes (a "portfolio" position). The position penalty method is introduced to improve the convergence. An Open-AI gym-compatible hedge environment is introduced and the Open AI SAC baseline RL engine is being used as a learning baseline.

## Recommendations
Model sized larger than 1024 x 1024 were trained on Amazon EC2 g4dn.2xlarge machine with the use of Deep Learning AMI (Ubuntu 16.04) Version 30.0 (ami-02379288a3b4cbe7b). On the AMI instance the latest CUDA-powered Pytorch version was used:
source activate pytorch_latest_p36

To install the libs as a module:
'''
cd stable-baselines3-autohedger-single
pip install .
'''

## Learning process
Learning progress is interactively displayed on a user machine with matplotlib plots for rewards and the dashboard. To monitor the learning process on a remote machine the reward figures as well as dashboards are also saved under directories autohedger-single/figs and autohedger-portfolio/figs. 
