import os
import tensorflow as tf
print(tf.__version__)
import numpy as np
from tensorflow import random_uniform_initializer


class OUActionNoise(object):
    # https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/DDPG
    # Creates temporally correlated noise with random normal distribution defined by:
    # theta (time), mu (mean), sigma (s.d.) and a previous value x_prev
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()  # sets previous value for noise

    def __call__(self):
        #  lets you get noise from object
        #  noise = OUActionNoise()  to instantiate object
        #  mynoise = noise() to get the noise
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)







