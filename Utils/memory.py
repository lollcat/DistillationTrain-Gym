# -*- coding: utf-8 -*-
"""
Taken from
https://github.com/udacity/deep-learning/blob/master/reinforcement/Q-learning-cart.ipynb
"""
from collections import deque
import numpy as np


class Memory:
    """
Taken from
https://github.com/udacity/deep-learning/blob/master/reinforcement/Q-learning-cart.ipynb
"""
    def __init__(self, max_size=100):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]
