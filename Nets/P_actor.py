from tensorflow.keras.layers import Dense, Input, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam



class ParameterAgent:
    def __init__(self, lr, n_continuous_actions, state_shape, layer_size=128):
        self.lr = lr
        self.n_continuous_actions = n_continuous_actions
        self.layer_size = layer_size
        self.state_shape = state_shape


    def build_network(self):
        input_state = Input(shape=self.state_shape, name="input_state")
        flat_input_state = Flatten(name="flat_input_state")(input_state)
        dense1 = Dense(self.layer_size, activation='relu', name="dense1")(flat_input_state)
        dense2 = Dense(self.layer_size, activation='relu', name="dense2")(dense1)
        dense3 = Dense(self.layer_size, activation='relu', name="dense3")(dense2)
        output = Dense(self.n_continuous_actions, activation='tanh', name="output")(dense3)

        model = Model(inputs=input_state, outputs=output)
        #optimizer = RMSprop(lr=self.lr)
        optimizer = Adam(lr=self.lr)

        return model, optimizer