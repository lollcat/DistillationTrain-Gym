from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float32')


class GaussianPolicy(Model):
    def __init__(self, n_actions, layer_size=256, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.layer1 = Dense(layer_size, activation='relu', name="dense1")
        self.layer2 = Dense(layer_size, activation='relu', name="dense2")
        self.layer3 = Dense(layer_size, activation='relu', name="dense3")

        self.mean = Dense(n_actions, activation='linear', name="mean")
        self.log_std = Dense(n_actions, activation='linear', name="std")


    def call(self, inputs):
        out = self.layer1(inputs)
        out = self.layer2(out)
        out = self.layer3(out)

        mean = self.mean(out)
        log_std = self.log_std(out)
        log_std = tf.keras.backend.clip(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample_action(self, state, epsilon=1e-6):
        mean, log_std = self.call(state)
        std = tf.math.exp(log_std)

        normal = tfp.distributions.MultivariateNormalDiag(mean, std)
        z = normal.sample()
        action = tf.math.tanh(z)
        log_pi = normal.log_prob(z) - tf.reduce_sum(tf.math.log(1 - tf.math.square(action) + epsilon), axis=1)
        log_pi = tf.expand_dims(log_pi, axis=1)
        return action, log_pi
