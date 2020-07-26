import numpy as np


class Tester:
    def __init__(self, param_model, dqn_model, env, product_all=False):
        self.product_all = product_all
        self.param_model = param_model
        self.dqn_model = dqn_model
        self.env = env
        self.n_discrete_actions = env.discrete_action_space.n
        self.allow_submit = env.allow_submit

    def test(self, render=False):
        state = self.env.reset()
        done = False
        score = 0
        while not done:
            state = state[np.newaxis, :]
            continuous_action = self.param_model.predict(state)
            predict_discrete = self.dqn_model.predict([state, continuous_action])
            illegal_actions = self.illegal_actions(state)
            predict_discrete[:, illegal_actions] = predict_discrete.min() - 1
            action_discrete = np.argmax(predict_discrete)

            state, reward, done, _ = self.env.step([continuous_action[0], action_discrete])
            score += reward
            if render is True:
                self.env.render()
        return self.env


    def illegal_actions(self, state):
        LK_legal1 = state[:, :, 0:-1] == 0
        LK_legal1 = LK_legal1.flatten(order="C")
        LK_legal2 = state[:, :, 1:] == 0
        LK_legal2 = LK_legal2.flatten(order="C")
        LK_legal = LK_legal1 + LK_legal2
        if self.product_all is True:
            non_explore_actions = np.zeros((self.n_discrete_actions,), dtype=bool)
            prev_LKs = self.env.sep_order
            for i in range(len(prev_LKs)):
                non_explore_actions[np.arange(self.n_discrete_actions) % 5 == prev_LKs[i]] = True
            LK_legal = LK_legal + non_explore_actions

        elif self.allow_submit is True:
            if self.env.n_outlet_streams > 1:
               LK_legal = np.append(LK_legal, False)
            else:
                LK_legal = np.append(LK_legal, True)

        return LK_legal
