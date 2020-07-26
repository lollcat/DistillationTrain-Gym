import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import clone_model

class Step:  # Stores a step
    def __init__(self, state, action_continuous, action_discrete, revenue, TAC, tops_state, bottoms_state, done):
        self.state = state
        self.action_continuous = action_continuous
        self.action_discrete = action_discrete
        self.revenue = revenue
        self.TAC = TAC
        self.tops_state = tops_state
        self.bottoms_state = bottoms_state
        self.done = done


class Worker_reward:
    def __init__(self, name, global_network_P, global_network_dqn, global_optimizer_P, global_optimizer_dqn,
                 global_counter, env, env_args, n_continuous_actions, max_global_steps, returns_list, summary_writer,
                 n_steps=10, gamma=0.999):
        self.name = name
        self.global_network_P = global_network_P
        self.global_network_dqn = global_network_dqn
        self.global_optimizer_P = global_optimizer_P
        self.global_optimizer_dqn = global_optimizer_dqn
        self.global_counter = global_counter
        self.env = env
        self.env_args = env_args
        self.state = False  # add state when we run
        self.max_global_steps = max_global_steps
        self.global_step = 0
        self.returns_list = returns_list
        self.n_steps = n_steps
        self.gamma = gamma
        self.start_time = time.time()

        self.local_param_model = clone_model(global_network_P)
        self.local_param_model.set_weights(global_network_P.get_weights())
        self.local_dqn_model = clone_model(global_network_dqn)
        self.local_dqn_model.set_weights(global_network_dqn.get_weights())

        self.summary_writer = summary_writer

    def run(self, coord):
        try:
            self.env = self.env(*self.env_args)
            self.state = self.env.reset()
            while not coord.should_stop():
                # Collect some experience
                experience = self.run_n_steps()
                # Update the global networks using local gradients
                self.update_global_parameters(experience)
                # Stop once the max number of global steps has been reached
                if self.max_global_steps is not None and self.global_step >= self.max_global_steps:
                    coord.request_stop()
                    return f'worker {self.name}, step: {self.global_step}'

        except tf.errors.CancelledError:
            print(f'worker {self.name} tf.errors.CancelledError')

    def choose_action(self, state, current_step, stop_step):
        state = state[np.newaxis, :]
        # get continuous action
        action_continuous = np.clip(self.local_param_model.predict(state) + 0.3 *
                                            np.random.normal(0, 1, size=self.env.continuous_action_space.shape[0]),
                                            a_min=-1, a_max=1)
        # get discrete action
        Q_value = np.sum(self.local_dqn_model.predict([state, action_continuous]))
        action_discrete = self.eps_greedy(Q_value, current_step, stop_step)

        action_continuous = action_continuous[0]  # take it back to the correct shape
        return action_continuous, action_discrete

    def eps_greedy(self, Q_value, current_step, stop_step, max_prob=1, min_prob=0):
        explore_threshold = max(max_prob - current_step / stop_step * (max_prob - min_prob), min_prob)
        random = np.random.rand()
        if random < explore_threshold:  # explore probabilities to bias in direction of seperating a stuff
            action_discrete = np.random.choice(a=[0, 1], p=[0.7, 0.3])
        else:  # exploit
            if Q_value > 0:  #  seperate
                action_discrete = 0
            else:  # submit
                action_discrete = 1
        return action_discrete

    def run_n_steps(self):
        experience = []
        score = 0
        for _ in range(self.n_steps):
            action = self.choose_action(self.state, self.global_step, round(self.max_global_steps))
            action_continuous, action_discrete = action
            with self.summary_writer.as_default():
                tf.summary.scalar('n_stages', action_continuous[0], step=self.global_step)
                tf.summary.scalar('reflux', action_continuous[1], step=self.global_step)
                tf.summary.scalar('reboil', action_continuous[2], step=self.global_step)
                tf.summary.scalar('pressure drop ratio', action_continuous[3], step=self.global_step)

            self.state = self.env.State.state.copy()
            tops_state, bottoms_state, revenue, TAC, done, info = self.env.step(action)
            if action_discrete == 0:  # seperating action
                step = Step(self.state, action_continuous, action_discrete, revenue, TAC, tops_state, bottoms_state,
                     done)
                experience.append(step)
            reward = revenue - TAC
            score += reward
            self.global_step = next(self.global_counter)

            if done:
                self.returns_list.append(score)
                with self.summary_writer.as_default():
                    tf.summary.scalar('total score', self.returns_list[-1], step=self.global_step)
                print(f"Worker: {self.name} Score is {score}, global steps {self.global_step}/{self.max_global_steps}")
                self.state = self.env.reset()
                break
        return experience

    #@tf.function
    def update_global_parameters(self, experience):
        with tf.device('/CPU:0'):
            accumulated_param_gradients = 0
            accumulated_dqn_gradients = 0
            for step in experience:
                gradient_param, gradient_dqn = self.get_gradient(step.state, step.action_continuous, step.revenue,
                                                                 step.TAC, step.tops_state, step.bottoms_state)
                if accumulated_dqn_gradients == 0:
                    accumulated_param_gradients = gradient_param
                    accumulated_dqn_gradients = gradient_dqn
                else:
                    accumulated_param_gradients = [tf.add(accumulated_param_gradients[i], gradient_param[i])
                                                   for i in range(len(gradient_param))]
                    accumulated_dqn_gradients = [tf.add(accumulated_dqn_gradients[i], gradient_dqn[i])
                                                 for i in range(len(gradient_dqn))]
            self.update_all_weights(accumulated_dqn_gradients, accumulated_param_gradients)
            return

    def get_gradient(self, state, action_continuous, revenue, TAC, tops_state, bottoms_state):
        # param part
        state = state[np.newaxis, :]
        tops_state = tops_state[np.newaxis, :]
        bottoms_state = bottoms_state[np.newaxis, :]
        action_continuous = action_continuous[np.newaxis, :]

        # next state includes tops and bottoms
        next_continuous_action_tops = self.local_param_model(tops_state)
        next_continuous_action_bottoms = self.local_param_model(bottoms_state)
        Q_value_top = tf.keras.backend.sum(self.local_dqn_model(
            [tops_state, next_continuous_action_tops]))
        Q_value_bottoms = tf.keras.backend.sum(self.local_dqn_model(
            [bottoms_state, next_continuous_action_bottoms]))
        # target
        # note we don't use the actual done values because the max function is doing a version of this
        target_next_state_value = self.gamma * (tf.math.maximum(Q_value_top, 0) + tf.math.maximum(Q_value_bottoms, 0))
        with tf.GradientTape(persistent=True) as tape:

            tape.watch(self.local_param_model.trainable_weights)
            predict_param = self.local_param_model(state)
            Q_value = tf.keras.backend.sum(self.local_dqn_model([state, predict_param]))
            loss_param = - Q_value

            # Q_net part
            tape.watch(self.local_dqn_model.trainable_weights)
            # compute Q net updates
            # first for TAC and revenue which is simple
            revenue_prediction, TAC_prediction, future_reward_prediction = \
                self.local_dqn_model([state, action_continuous])
            loss_revenue = tf.keras.losses.MSE(revenue_prediction,
                                               tf.convert_to_tensor(revenue, dtype=np.float32))
            loss_TAC = tf.keras.losses.MSE(TAC_prediction, tf.convert_to_tensor(TAC, dtype=np.float32))
            loss_next_state_value = tf.keras.losses.MSE(tf.convert_to_tensor(target_next_state_value, np.float32),
                                                        future_reward_prediction)

        gradient_param = tape.gradient(loss_param, self.local_param_model.trainable_weights)
        gradient_next_state_value = tape.gradient(loss_next_state_value, self.local_dqn_model.trainable_weights,
                                                  unconnected_gradients=tf.UnconnectedGradients.ZERO)  # not including revenue and loss
        gradient_revenue = tape.gradient(loss_revenue, self.local_dqn_model.trainable_weights,
                                         unconnected_gradients=tf.UnconnectedGradients.ZERO)
        gradient_TAC = tape.gradient(loss_TAC, self.local_dqn_model.trainable_weights,
                                     unconnected_gradients=tf.UnconnectedGradients.ZERO)
        gradient_dqn_total = [dqn_grad + gradient_TAC[i] + gradient_revenue[i]
                              for i, dqn_grad in enumerate(gradient_next_state_value)]

        with self.summary_writer.as_default():
            tf.summary.scalar('param loss', loss_param, step=self.global_step)
            tf.summary.scalar('loss_next_state_value', loss_next_state_value[0], step=self.global_step)
            tf.summary.scalar('revenue loss', loss_revenue[0], step=self.global_step)
            tf.summary.scalar('TAC loss', loss_TAC[0], step=self.global_step)
            tf.summary.scalar('DQN TOTAL loss', loss_TAC[0] + loss_revenue[0] + loss_next_state_value[0], step=self.global_step)
        return gradient_param, gradient_dqn_total

    def update_all_weights(self, accumulated_dqn_gradients, accumulated_param_gradients):
        self.global_optimizer_dqn.apply_gradients(zip(accumulated_dqn_gradients,
                                                      self.global_network_dqn.trainable_weights))
        self.local_dqn_model.set_weights(self.global_network_dqn.get_weights())
        if accumulated_param_gradients is 0:
            return
        else:
            # update global nets
            self.global_optimizer_P.apply_gradients(zip(accumulated_param_gradients,
                                                        self.global_network_P.trainable_weights))
            # update local nets
            self.local_param_model.set_weights(self.global_network_P.get_weights())
            return
