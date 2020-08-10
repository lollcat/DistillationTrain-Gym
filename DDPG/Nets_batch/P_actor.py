from tensorflow.keras.layers import Dense, Input, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



class ParameterAgent:
    def __init__(self, lr, n_continuous_actions, state_shape, layer_size=512):
        self.lr = lr
        self.n_continuous_actions = n_continuous_actions
        self.layer_size = layer_size
        self.state_shape = state_shape


    def build_network(self):
        input_state = Input(shape=(self.state_shape[1], ), name="input_state")
        out = Dense(self.layer_size, activation='relu', name="dense1")(input_state)
        out = BatchNormalization()(out)
        out = Dense(self.layer_size, activation='relu', name="dense2")(out)
        out = BatchNormalization()(out)
        out = Dense(self.layer_size, activation='relu', name="dense3")(out)
        out = BatchNormalization()(out)
        output = Dense(self.n_continuous_actions, activation='tanh', name="output")(out)

        model = Model(inputs=input_state, outputs=output)
        optimizer = Adam(lr=self.lr)

        return model, optimizer