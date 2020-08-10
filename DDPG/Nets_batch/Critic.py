from tensorflow.keras.layers import Dense, Input, Concatenate, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class Critic:
    def __init__(self, lr, n_continuous_actions, state_shape, layer_size=512):
        self.lr = lr
        self.n_continuous_actions = n_continuous_actions
        self.layer_size = layer_size
        self.state_shape = state_shape


    def build_network(self):
        input_state = Input(shape=(self.state_shape[1],), name="input_state")
        state_out = Dense(16, activation='relu')(input_state)
        state_out = BatchNormalization()(state_out)
        state_out = Dense(32, activation='relu')(state_out)
        state_out = BatchNormalization()(state_out)

        input_parameters = Input(shape=(self.n_continuous_actions,), name="input_parameters")
        action_out = Dense(32, activation='relu')(input_parameters)
        action_out = BatchNormalization()(action_out)

        concat = Concatenate(name="concat")([state_out, action_out])

        out = Dense(self.layer_size, activation='relu', name="dense1")(concat)
        out = BatchNormalization()(out)
        out = Dense(self.layer_size, activation= 'relu', name="dense2")(out)
        out = BatchNormalization()(out)
        out = Dense(self.layer_size, activation='relu', name="dense3")(out)
        out = BatchNormalization()(out)

        future_reward = Dense(1, activation='linear', name="future_reward")(out)
        revenue = Dense(1, activation='linear', name="revenue")(out)
        cost = Dense(1, activation='linear', name="TAC")(out)

        model = Model(inputs=[input_state, input_parameters], outputs=[future_reward, revenue, cost])
        optimizer = Adam(lr=self.lr)

        return model, optimizer