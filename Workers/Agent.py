from Utils.memory import Memory
from Nets_batch.Critic import Critic
from Nets_batch.P_actor import ParameterAgent
from Env.DC_gym_reward import DC_gym_reward as DC_Gym
from Env.STANDARD_CONFIG import CONFIG
import numpy as np
from tensorflow.keras.models import clone_model
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
standard_args = CONFIG(1).get_config()
class Agent:
    def __init__(self, summary_writer, total_episodes=500, env=DC_Gym(*standard_args), actor_lr=0.0001, critic_lr=0.001,
                 batch_size=32, mem_length=1000, gamma=0.99, tau=0.001):
        assert batch_size < mem_length
        self.total_episodes = total_episodes
        self.env = env
        self.batch_size = batch_size
        self.memory = Memory(max_size=mem_length)
        self.param_model, self.param_optimizer = \
            ParameterAgent(actor_lr, env.continuous_action_space.shape[0], env.observation_space.shape).build_network()
        self.critic_model, self.critic_optimizer = \
            Critic(critic_lr, env.continuous_action_space.shape[0],
                      env.observation_space.shape).build_network()
        self.param_model_target = clone_model(self.param_model)
        self.param_model_target.set_weights(self.param_model.get_weights())
        self.critic_model_target = clone_model(self.critic_model)
        self.critic_model_target.set_weights(self.critic_model.get_weights())
        self.gamma = gamma
        self.history = []
        self.summary_writer = summary_writer
        self.step = 0
        self.tau = tau

    def update_target_networks(self):
        self.param_model_target.set_weights([tf.math.multiply(local_weight, self.tau) +
                                    tf.math.multiply(target_weight, 1-self.tau)
                                      for local_weight, target_weight in
                                      zip(self.param_model.get_weights(), self.param_model_target.get_weights())])
        self.critic_model_target.set_weights([tf.math.multiply(local_weight, self.tau) +
                                    tf.math.multiply(target_weight, 1-self.tau)
                                      for local_weight, target_weight in
                                      zip(self.critic_model.get_weights(), self.critic_model_target.get_weights())])


    def run_episodes(self):
        for i in range(self.total_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                self.step += 1
                state = self.env.State.state.copy()
                action_continuous = np.clip(self.param_model.predict(state[np.newaxis, :]) + 0.3 *
                                            np.random.normal(0, 1, size=self.env.continuous_action_space.shape[0]),
                                            a_min=-1, a_max=1)
                Q_value = np.sum(self.critic_model.predict([state[np.newaxis, :], action_continuous]), axis=0)
                if state[:, 0: self.env.n_components].max()*self.env.State.flow_norm <= self.env.min_total_flow*1.1:
                    # must submit if there is not a lot of flow, add bit of extra margin to prevent errors
                    action_discrete = 1
                else:
                    action_discrete = self.eps_greedy(Q_value, i, round(self.total_episodes*3/4))
                action_continuous = action_continuous[0]
                action = action_continuous, action_discrete
                # now take action
                tops_state, bottoms_state, annual_revenue, TAC, done, info = self.env.step(action)
                with self.summary_writer.as_default():
                    tf.summary.scalar('n_stages', action_continuous[0], step=self.step)
                    tf.summary.scalar('reflux', action_continuous[1], step=self.step)
                    tf.summary.scalar('reboil', action_continuous[2], step=self.step)
                    tf.summary.scalar('pressure drop ratio', action_continuous[3], step=self.step)
                    tf.summary.scalar('TAC', TAC, step=self.step)
                    tf.summary.scalar('revenue', annual_revenue, step=self.step)
                reward = annual_revenue + TAC
                total_reward += reward
                if action_discrete == 0:  # seperating action
                    mass_balance_rel_error = np.absolute((state[:, 0:self.env.n_components] - (
                                tops_state[:, 0:self.env.n_components] +
                                bottoms_state[:, 0:self.env.n_components]))
                                             / np.maximum(state[:, 0:self.env.n_components], 0.1)) # 0 to prevent divide by 0
                    assert mass_balance_rel_error.max() < 0.05, \
                        f"Max error: {mass_balance_rel_error.max()}"\
                        f"MB rel error: {mass_balance_rel_error}" \
                        f"(state: {state[:, 0:self.env.n_components]}" \
                                                              f"tops: {tops_state[:, 0:self.env.n_components]}" \
                                                              f"bottoms: {bottoms_state[:, 0:self.env.n_components]}"
                    self.memory.add(
                        (state, action_continuous, action_discrete, annual_revenue, TAC, tops_state, bottoms_state,
                         1 - done))
                if len(self.memory.buffer) > self.batch_size:
                    self.learn()

            with self.summary_writer.as_default():
                tf.summary.scalar('total score', total_reward, step=i)
            self.history.append(total_reward)


            if i % (self.total_episodes/10) == 0:
                print(f"Average reward of last {round(self.total_episodes/10)} episodes: "
                      f"{np.mean(self.history[-round(self.total_episodes/10):])}")
                print(f"episode {i}/{self.total_episodes}")

    def learn(self):
        # now sample from batch & train
        batch = self.memory.sample(self.batch_size)
        state_batch = np.array([each[0] for each in batch])
        continuous_action_batch = np.array([each[1] for each in batch])
        discrete_action_batch = np.array([each[2] for each in batch])
        annual_revenue_batch = np.array([each[3] for each in batch]).reshape(self.batch_size, 1)
        TAC_batch = np.array([each[4] for each in batch]).reshape(self.batch_size, 1)
        tops_state_batch = np.array([each[5] for each in batch])
        bottoms_state_batch = np.array([each[6] for each in batch])
        done_batch = np.array([each[7] for each in batch])

        # next state includes tops and bottoms
        next_continuous_action_tops = self.param_model_target.predict_on_batch(tops_state_batch)
        next_continuous_action_bottoms = self.param_model_target.predict_on_batch(bottoms_state_batch)
        Q_value_top = tf.keras.backend.sum(self.critic_model_target.predict_on_batch(
            [tops_state_batch, next_continuous_action_tops]), axis=0)
        Q_value_bottoms = tf.keras.backend.sum(self.critic_model_target.predict_on_batch(
            [bottoms_state_batch, next_continuous_action_bottoms]), axis=0)
        # target
        # note we don't use the actual done values because the max function is doing a version of this
        target_next_state_value = self.gamma * (tf.math.maximum(Q_value_top, 0) + tf.math.maximum(Q_value_bottoms, 0))
        # double check if need be
        mass_balance_rel_error_max = ((state_batch[:, :, 0:self.env.n_components] - (
                tops_state_batch[:, :, 0:self.env.n_components] +
                bottoms_state_batch[:, :, 0:self.env.n_components]))
                                 / np.maximum(state_batch[:, :, 0:self.env.n_components], 0.1)).max() # min to prevent divide by 0
        assert mass_balance_rel_error_max < 0.05
        with tf.GradientTape() as tape0, tf.GradientTape(persistent=True) as tape1:
            tape0.watch(self.param_model.trainable_weights)
            tape1.watch(self.critic_model.trainable_weights)
            predict_param = self.param_model.predict_on_batch(state_batch)
            critic_value = tf.keras.backend.sum(self.critic_model.predict_on_batch([state_batch, predict_param]), axis=0)
            loss_param = - tf.math.reduce_mean(critic_value)

            # compute Q net updates
            # first for TAC and revenue which is simple
            revenue_prediction, TAC_prediction, future_reward_prediction = \
                self.critic_model.predict_on_batch([state_batch, continuous_action_batch])
            loss_revenue = tf.math.reduce_mean(tf.keras.losses.MSE(tf.convert_to_tensor(annual_revenue_batch, dtype=np.float32),
                                               revenue_prediction))
            loss_TAC = tf.math.reduce_mean(tf.keras.losses.MSE(tf.convert_to_tensor(TAC_batch, dtype=np.float32),
                                                               TAC_prediction))
            loss_next_state_value = tf.math.reduce_mean(tf.keras.losses.MSE(
                tf.convert_to_tensor(target_next_state_value, np.float32), future_reward_prediction))

        with self.summary_writer.as_default():
            tf.summary.scalar("param batch mean loss", loss_param, step=self.step)
            tf.summary.scalar("revenue batch mean loss", loss_revenue, step=self.step)
            tf.summary.scalar("TAC batch mean loss", loss_TAC, step=self.step)
            tf.summary.scalar("next_state_value batch mean loss", loss_next_state_value, step=self.step)
            tf.summary.scalar("total DQN loss", loss_TAC + loss_revenue + loss_next_state_value, step=self.step)

        # get gradients of loss with respect to the param_model weights
        gradient_param = tape0.gradient(loss_param, self.param_model.trainable_weights)
        gradient_next_state_value = tape1.gradient(loss_next_state_value, self.critic_model.trainable_weights,
                                     unconnected_gradients=tf.UnconnectedGradients.ZERO)  # not including revenue and loss
        gradient_revenue = tape1.gradient(loss_revenue, self.critic_model.trainable_weights,
                                         unconnected_gradients=tf.UnconnectedGradients.ZERO)
        gradient_TAC = tape1.gradient(loss_TAC, self.critic_model.trainable_weights,
                                     unconnected_gradients=tf.UnconnectedGradients.ZERO)
        gradient_dqn_total = [dqn_grad + gradient_TAC[i] + gradient_revenue[i]
                              for i, dqn_grad in enumerate(gradient_next_state_value)]

        # update global parameters
        self.param_optimizer.apply_gradients(zip(gradient_param, self.param_model.trainable_weights))
        self.critic_optimizer.apply_gradients(zip(gradient_dqn_total, self.critic_model.trainable_weights))
        self.update_target_networks()

    def eps_greedy(self, Q_value, current_step, stop_step, max_prob=1, min_prob=0.05):
        if self.env.current_step is 0:  # must at least seperate first stream
            return 0
        else:
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

    def test_run(self):
        state = self.env.reset()
        done = False
        total_reward = 0
        i = 0
        while not done:
            i +=1
            state = self.env.State.state.copy()
            action_continuous = self.param_model.predict(state[np.newaxis, :])
            Q_value = np.sum(self.critic_model.predict([state[np.newaxis, :], action_continuous]), axis=0)
            if state[:, 0: self.env.n_components].max() * self.env.State.flow_norm <= self.env.min_total_flow * 1.1:
                # must submit if there is not a lot of flow, add bit of extra margin to prevent errors
                action_discrete = 1
            else:
                if Q_value > 0:
                    action_discrete = 0
                else:
                    action_discrete = 1
            action_continuous = action_continuous[0]
            action = action_continuous, action_discrete
            # now take action
            tops_state, bottoms_state, annual_revenue, TAC, done, info = self.env.step(action)
            reward = annual_revenue + TAC
            total_reward += reward
            print(f"step {i}: \n annual_revenue: {annual_revenue}, TAC: {TAC} \n Q_value {Q_value}, reward {reward}")
