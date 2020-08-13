from Utils.memory import Memory
from DDPG.Nets_batch.Critic import Critic
from DDPG.Nets_batch.P_actor import ParameterAgent
from Env.DC_gym import DC_Gym
from Env.STANDARD_CONFIG import CONFIG
import numpy as np
from tensorflow.keras.models import clone_model
import tensorflow as tf
from DDPG.Utils.OrnsteinNoise import OUActionNoise
import pickle
import time

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
standard_args = CONFIG(1).get_config()
class Agent:
    def __init__(self, summary_writer, total_episodes=500, env=DC_Gym(*standard_args, simple_state=True), actor_lr=0.0001, critic_lr=0.001,
                 batch_size=32, mem_length=1e4, min_memory_length=1e3, gamma=0.99, tau=0.001, use_old_memories=False):
        assert batch_size < mem_length
        assert min_memory_length <= mem_length
        self.total_episodes = total_episodes
        self.env = env
        self.batch_size = batch_size
        self.memory = Memory(max_size=int(mem_length))
        self.min_memory_length = int(min_memory_length)
        self.use_load_memory = use_old_memories
        if use_old_memories is True:
            self.load_memory()
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
        self.noise = OUActionNoise(mu=np.zeros(env.continuous_action_space.shape[0]))

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
        if self.use_load_memory is True:
            self.load_memory()
        else:
            self.fill_memory()
        for i in range(self.total_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            current_step = 0
            while not done:
                self.step += 1
                current_step += 1
                state = self.env.State.state.copy()
                #noise = self.noise()
                noise = 0.3 * np.random.normal(0, 1, size=(1, self.env.continuous_action_space.shape[0]))
                action_continuous = np.clip(self.param_model.predict(state) + noise,
                                            a_min=-1, a_max=1)
                Q_value = np.sum(self.critic_model.predict([state, action_continuous]), axis=0)
                if state[:, 0: self.env.n_components].max()*self.env.State.flow_norm <= self.env.min_total_flow*1.1:
                    # must submit if there is not a lot of flow, add bit of extra margin to prevent errors
                    action_discrete = 1
                else:
                    #action_discrete = self.eps_greedy(Q_value, i, round(self.total_episodes*3/4))
                    if Q_value > 0:
                        action_discrete = 0
                    else:
                        action_discrete = 1
                action_continuous = action_continuous[0]
                action = action_continuous, action_discrete
                # now take action
                next_state, annual_revenue, TAC, done, info = self.env.step(action)
                tops_state, bottoms_state = next_state
                reward = annual_revenue + TAC
                total_reward += reward

                if action_discrete == 0:  # seperating action
                    if len(info) == 2:  # this means we didn't have a failed solve
                        # note we scale the reward here
                        self.memory.add(
                            (state, action_continuous, annual_revenue, TAC, tops_state, bottoms_state,
                             1 - info[0], 1 - info[1]))
                        with self.summary_writer.as_default():
                            tf.summary.scalar('n_stages', action_continuous[0], step=self.step)
                            tf.summary.scalar('reflux', action_continuous[1], step=self.step)
                            tf.summary.scalar('reboil', action_continuous[2], step=self.step)
                            tf.summary.scalar('pressure drop ratio', action_continuous[3], step=self.step)
                            tf.summary.scalar('TAC', TAC, step=self.step)
                            tf.summary.scalar('revenue', annual_revenue, step=self.step)
                self.learn()

            with self.summary_writer.as_default():
                tf.summary.scalar('total score', total_reward, step=i)
                tf.summary.scalar("episode length", current_step, step=i)
            self.history.append(total_reward)


            if i % (self.total_episodes/10) == 0:
                print(f"Average reward of last {round(self.total_episodes/10)} episodes: "
                      f"{np.mean(self.history[-round(self.total_episodes/10):])}")
                print(f"episode {i}/{self.total_episodes}")

        self.save_memory()

    def learn(self):
        # now sample from batch & train
        batch = self.memory.sample(self.batch_size)
        states = np.squeeze(np.array([each[0] for each in batch]))
        actions = np.array([each[1] for each in batch])
        annual_revenues = np.array([each[2] for each in batch]).reshape(self.batch_size, 1)
        TACs = np.array([each[3] for each in batch]).reshape(self.batch_size, 1)
        tops_states = np.squeeze(np.array([each[4] for each in batch]))
        bottoms_states = np.squeeze(np.array([each[5] for each in batch]))
        tops_dones = np.array([each[6] for each in batch]).reshape(self.batch_size, 1)
        bottoms_dones = np.array([each[7] for each in batch]).reshape(self.batch_size, 1)

        # next state includes tops and bottoms
        next_continuous_action_tops = self.param_model_target.predict_on_batch(tops_states)
        next_continuous_action_bottoms = self.param_model_target.predict_on_batch(bottoms_states)
        Q_value_top = tf.keras.backend.sum(self.critic_model_target.predict_on_batch(
            [tops_states, next_continuous_action_tops]), axis=0) * tops_dones
        Q_value_bottoms = tf.keras.backend.sum(self.critic_model_target.predict_on_batch(
            [bottoms_states, next_continuous_action_bottoms]), axis=0) * bottoms_dones
        # target
        # note we don't use the actual done values because the max function is doing a version of this
        target_next_state_value = self.gamma * (tf.math.maximum(Q_value_top, 0) + tf.math.maximum(Q_value_bottoms, 0))
        # double check if need be
        with tf.GradientTape() as tape0, tf.GradientTape(persistent=True) as tape1:
            tape0.watch(self.param_model.trainable_weights)
            tape1.watch(self.critic_model.trainable_weights)
            predict_param = self.param_model.predict_on_batch(states)
            critic_value = tf.keras.backend.sum(self.critic_model.predict_on_batch([states, predict_param]), axis=0)
            loss_param = - tf.math.reduce_mean(critic_value)

            # compute Q net updates
            # first for TAC and revenue which is simple
            revenue_prediction, TAC_prediction, future_reward_prediction = \
                self.critic_model.predict_on_batch([states, actions])
            loss_revenue = tf.math.reduce_mean(tf.keras.losses.MSE(tf.convert_to_tensor(annual_revenues, dtype=np.float32),
                                               revenue_prediction))
            loss_TAC = tf.math.reduce_mean(tf.keras.losses.MSE(tf.convert_to_tensor(TACs, dtype=np.float32),
                                                               TAC_prediction))
            loss_next_state_value = tf.math.reduce_mean(tf.keras.losses.MSE(
                tf.convert_to_tensor(target_next_state_value, np.float32), future_reward_prediction))

        with self.summary_writer.as_default():
            tf.summary.scalar("param batch mean loss", loss_param, step=self.step)
            tf.summary.scalar("revenue batch mean loss", loss_revenue, step=self.step)
            tf.summary.scalar("TAC batch mean loss", loss_TAC, step=self.step)
            tf.summary.scalar("next_state_value batch mean loss", loss_next_state_value, step=self.step)
            tf.summary.scalar("total DQN loss", loss_TAC + loss_revenue + loss_next_state_value, step=self.step)
            tf.summary.scalar("next_state_value_target", tf.reduce_mean(target_next_state_value), step=self.step)

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

    def eps_greedy(self, Q_value, current_step, stop_step, max_prob=0.5, min_prob=0.01):
        if self.env.current_step is 0:  # must at least seperate first stream
            return 0
        else:
            explore_threshold = max(max_prob - current_step / stop_step * (max_prob - min_prob), min_prob)
            random = np.random.rand()
            if random < explore_threshold:
                action_discrete = np.random.choice(a=[0, 1], p=[0.5, 0.5])
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
            i += 1
            state = self.env.State.state.copy()
            action_continuous = self.param_model.predict(state[np.newaxis, :])
            Q_values = self.critic_model.predict([state[np.newaxis, :], action_continuous])
            Q_value = np.sum(Q_values, axis=0)
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
            next_state, annual_revenue, TAC, done, info = self.env.step(action)
            tops_state, bottoms_state = next_state
            reward = annual_revenue + TAC
            total_reward += reward
            print(f"step {i}: \n annual_revenue: {annual_revenue}, TAC: {TAC} \n Q_values {Q_values}, reward {reward}")

    def save_memory(self):
        pickle.dump(self.memory, open("./DDPG/memory_data/" + str(time.time()) + ".obj", "wb"))
        pickle.dump(self.memory, open("./DDPG/memory_data/memory.obj", "wb"))

    def load_memory(self):

        """
        Currently use load_memory to skip having to fill memory with random experiences
        Could do something with below to run without running the env to save time
        "./DDPG/memory_data/memory.obj"
        """
        old_memory = pickle.load(open("./DDPG/memory_data/random_memory.obj", "rb"))
        self.memory.buffer += old_memory.buffer

    def fill_memory(self):
        done = False
        state = self.env.reset()
        while len(self.memory.buffer) < self.min_memory_length:
            state = self.env.State.state.copy()
            action_continuous, _ = self.env.sample()
            if state[:, 0: self.env.n_components].max() * self.env.State.flow_norm <= self.env.min_total_flow * 1.1:
                # must submit if there is not a lot of flow, add bit of extra margin to prevent errors
                action_discrete = 1
            else:
                action_discrete = np.random.choice([0, 1], p=[0.9, 0.1])
            action = action_continuous, action_discrete
            next_state, annual_revenue, TAC, done, info = self.env.step(action)
            tops_state, bottoms_state = next_state
            reward = annual_revenue + TAC  # TAC's sign is included in the env

            if action_discrete == 0:
                    if len(info) == 2:  # this means we didn't have a failed solve
                        # note we scale the reward here
                        self.memory.add((state, action_continuous, annual_revenue, TAC, tops_state, bottoms_state,
                                         1 - info[0], 1 - info[1]))
                        if len(self.memory.buffer) % (self.min_memory_length/10) == 0:
                            print(f"memory {len(self.memory.buffer)}/{self.min_memory_length}")
            if done is True:
                state = self.env.reset()

        pickle.dump(self.memory, open("./DDPG/memory_data/random_memory.obj", "wb"))