# Autohedging / market-making with reinforcement learning
This code repository supports the paper: https://arxiv.org/pdf/2008.12275.pdf

## Abstract
The paper explores the application of a continuous action space soft actor-critic (SAC) reinforcement learning model to the area of automated market-making. The reinforcement learning agent receives a simulated flow of client trades, thus accruing a position in an asset, and learns to offset this risk by either hedging at simulated "exchange" spreads or by attracting an offsetting client flow by changing offered client spreads (skewing the offered prices). The question of learning minimum spreads that compensate for the risk of taking the position is being investigated. Finally, the agent is posed with a problem of learning to hedge a blended client trade flow resulting from independent price processes (a "portfolio" position). The position penalty method is introduced to improve the convergence. An Open-AI gym-compatible hedge environment is introduced and the Open AI SAC baseline RL engine is being used as a learning baseline.

