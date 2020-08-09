from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float64')


class DeterministicPolicy(Model):
    def __init__(self, n_actions, layer_size=256):
        super(DeterministicPolicy, self).__init__()
        self.n_actions = n_actions
        self.layer1 = Dense(layer_size, activation='relu', name="dense1")
        self.layer2 = Dense(layer_size, activation='relu', name="dense2")
        self.layer3 = Dense(layer_size, activation='relu', name="dense3")

        self.mean = Dense(n_actions, activation='linear', name="mean")

    def call(self, inputs):
        out = self.layer1(inputs)
        out = self.layer2(out)
        out = self.layer3(out)
        mean = self.mean(out)
        action = tf.math.tanh(mean)
        return action

    def sample_action(self, state, noise_multiplier=0.3):
        mean = np.zeros((state.shape[0], self.n_actions))
        std = np.ones((state.shape[0], self.n_actions))
        noise = noise_multiplier * tfp.distributions.MultivariateNormalDiag(mean, std).sample()
        action = self.call(state)
        assert action.shape == noise.shape
        return action + noise
