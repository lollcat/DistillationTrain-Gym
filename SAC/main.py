"""
A bit of help from
https://towardsdatascience.com/in-depth-review-of-soft-actor-critic-91448aba63d4
"""
# tensorboard --logdir logs
from SAC_Agent.Agent import Agent
SAC = Agent(total_eps=500)
SAC.run()
