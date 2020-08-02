from tensorflow.keras.layers import Dense, Concatenate, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

class Critic(Model):
    def __init__(self, layer_size=256):
        super(Critic, self).__init__()
        self.state_layer_1 = Dense(16, activation='relu')
        self.state_layer_2 = Dense(32, activation='relu')

        self.action_layer = Dense(32, activation='relu')

        self.concat = Concatenate(name="concat")

        self.layer1 = Dense(layer_size, activation='relu', name="dense1")
        self.layer2 = Dense(layer_size, activation='relu', name="dense2")
        self.layer3 = Dense(layer_size, activation='relu', name="dense3")

        self.Q_value = Dense(1, activation='linear', name="Q_value")


    def call(self, inputs):
        input_state, input_actions = inputs

        state_out = self.state_layer_1(input_state)
        state_out = self.state_layer_2(state_out)

        action_out = self.action_layer(input_actions)

        concat = self.concat([state_out, action_out])

        out = self.layer1(concat)
        out = self.layer2(out)
        out = self.layer3(out)

        return self.Q_value(out)


