from Hard_Actor_Critic.Nets.Actor import DeterministicPolicy as Actor
from Hard_Actor_Critic.Nets.Critic import Critic
from Utils.memory import Memory
import tensorflow as tf
import numpy as np
import time
import pickle
from Env.DC_gym import DC_Gym
from Env.STANDARD_CONFIG import CONFIG
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
standard_args = CONFIG(1).get_config()


class Agent:
    """
    SAC without the soft
    """
    def __init__(self, env=DC_Gym(*standard_args, simple_state=True), total_eps=2e2, batch_size=64, alpha=0.2, max_mem_length=1e4, tau=0.005,
                 Q_lr=3e-3, policy_lr=3e-4, gamma=0.99, description="", use_load_memory=False,
                 actor_update_period=0, reward_scaling=1):
        self.env = env
        self.total_eps = int(total_eps)
        self.eps_greedy_stop_step = int(total_eps*3/4)
        self.steps = 0
        self.total_scores = []
        self.batch_size = batch_size
        self.tau = tau
        self.memory = Memory(int(max_mem_length))
        self.gamma = gamma

        self.Actor = Actor(env.real_continuous_action_space.shape[0])
        self.Q1 = Critic()
        self.Q2 = Critic()
        self.target_Q1 = Critic()
        self.target_Q1.set_weights(self.Q1.get_weights())
        self.target_Q2 = Critic()
        self.target_Q2.set_weights(self.Q2.get_weights())

        self.Q1_optimizer = tf.keras.optimizers.Adam(Q_lr)
        self.Q2_optimizer = tf.keras.optimizers.Adam(Q_lr)
        self.Actor_optimizer = tf.keras.optimizers.Adam(policy_lr)
        self.reward_scaling = reward_scaling

        if description == "":
            #description = f"Q_lr {Q_lr}, policy_lr {policy_lr}, alpha_lr {alpha_lr}"
            description = f"reward_scaling={reward_scaling}__"
        log_dir = './logs/HAC__' + description + \
                  time.asctime(time.localtime(time.time())).replace(" ", "_").replace(":", "-")
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.use_load_memory = use_load_memory

    def run(self):
        for ep in range(self.total_eps):
            total_score = 0
            done = False
            state = self.env.reset()
            current_step = 0
            while not done:
                current_step += 1
                self.steps += 1
                state = self.env.State.state.copy()
                action_continuous = self.Actor.sample_action(state)
                if state[:, 0: self.env.n_components].max() * self.env.State.flow_norm <= self.env.min_total_flow * 1.1:
                    # must submit if there is not a lot of flow, add bit of extra margin to prevent errors
                    action_discrete = 1
                else:
                    Q_value = tf.minimum(self.Q1([state, action_continuous]), self.Q2([state, action_continuous]))
                    action_discrete = self.get_discrete_action(Q_value)
                action_continuous = np.squeeze(action_continuous, axis=0)
                action = action_continuous, action_discrete
                next_state, annual_revenue, TAC, done, info = self.env.step(action)
                tops_state, bottoms_state = next_state
                reward = annual_revenue + TAC  # TAC's sign is included in the env
                total_score += reward

                if action_discrete == 0:
                        if len(info) == 2:  # this means we didn't have a failed solve
                            # note we scale the reward here
                            self.memory.add((state, action_continuous, reward*self.reward_scaling, tops_state, bottoms_state,
                                             1 - info[0], 1 - info[1]))
                            with self.summary_writer.as_default():
                                tf.summary.scalar('n_stages', action_continuous[0], step=self.steps)
                                tf.summary.scalar('reflux', action_continuous[1], step=self.steps)
                                tf.summary.scalar('reboil', action_continuous[2], step=self.steps)
                                tf.summary.scalar('pressure drop ratio', action_continuous[3], step=self.steps)
                                tf.summary.scalar('TAC', TAC, step=self.steps)
                                tf.summary.scalar('revenue', annual_revenue, step=self.steps)

                if len(self.memory.buffer) > self.batch_size:
                    self.learn()

            with self.summary_writer.as_default():
                tf.summary.scalar("total score", total_score, ep)
                tf.summary.scalar("episode length", current_step, ep)
            self.total_scores.append(total_score)
        self.save_memory()

    #@tf.function
    def learn(self):
        batch = self.memory.sample(self.batch_size)
        states = np.squeeze(np.array([each[0] for each in batch]))
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch]).reshape(self.batch_size, 1)
        tops_states = np.squeeze(np.array([each[3] for each in batch]))
        bottoms_states = np.squeeze(np.array([each[4] for each in batch]))
        tops_dones = np.array([each[5] for each in batch]).reshape(self.batch_size, 1)
        bottoms_dones = np.array([each[6] for each in batch]).reshape(self.batch_size, 1)

        tops_actions = self.Actor.sample_action(tops_states)
        bottoms_actions = self.Actor.sample_action(bottoms_states)
        tops_hardQ = tf.minimum(self.target_Q1([tops_states, tops_actions]), self.target_Q2([tops_states, tops_actions]))
        bottoms_hardQ = tf.minimum(self.target_Q1([bottoms_states, bottoms_actions]), self.target_Q2([bottoms_states, bottoms_actions]))
        # sum over new generated states to get value
        # neither can be 0 because then we would stop seperating, but bounding probs has no effect (Q generally > 0)
        next_Q_target = tops_dones * tf.maximum(tops_hardQ, 0) + \
                        bottoms_dones * tf.maximum(bottoms_hardQ, 0)  # soft target
        Q_expected = rewards + self.gamma * next_Q_target  # cannot be negative as then would separate
        assert Q_expected.shape == (self.batch_size, 1)
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            tape1.watch(self.Q1.trainable_variables)
            tape2.watch(self.Q2.trainable_variables)
            Q1_predict = self.Q1([states, actions])
            Q2_predict = self.Q2([states, actions])
            loss_Q1 = tf.reduce_mean((Q_expected - Q1_predict)**2)
            loss_Q2 = tf.reduce_mean((Q_expected - Q2_predict)**2)
        Q1_gradient = tape1.gradient(loss_Q1, self.Q1.trainable_variables)
        Q2_gradient = tape2.gradient(loss_Q2, self.Q2.trainable_variables)
        self.Q1_optimizer.apply_gradients(zip(Q1_gradient, self.Q1.trainable_variables))
        self.Q2_optimizer.apply_gradients(zip(Q2_gradient, self.Q2.trainable_variables))
        del tape1, tape2

        with tf.GradientTape() as actor_tape:
            actor_tape.watch(self.Actor.trainable_variables)
            predicted_actions= self.Actor.sample_action(states)
            Q1_predicted = self.Q1([states, predicted_actions])
            Q2_predicted = self.Q2([states, predicted_actions])
            Q_target = tf.minimum(Q1_predicted, Q2_predicted)
            actor_loss = tf.reduce_mean(- Q_target)
        actor_gradient = actor_tape.gradient(actor_loss, self.Actor.trainable_variables)
        self.Actor_optimizer.apply_gradients(zip(actor_gradient, self.Actor.trainable_variables))
        del actor_tape
        self.update_targets()

        with self.summary_writer.as_default():
            tf.summary.scalar("Q1 loss", loss_Q1, self.steps)
            tf.summary.scalar("Q2 loss", loss_Q2, self.steps)
            tf.summary.scalar("Actor loss", actor_loss, self.steps)
            tf.summary.scalar("actor_Q_target", tf.reduce_mean(Q_target), self.steps)

    #@tf.function
    def update_targets(self):
        self.target_Q1.set_weights([tf.math.multiply(local_weight, self.tau) +
                                    tf.math.multiply(target_weight, 1-self.tau)
                                      for local_weight, target_weight in
                                      zip(self.Q1.get_weights(), self.target_Q1.get_weights())])
        self.target_Q2.set_weights([tf.math.multiply(local_weight, self.tau) +
                                    tf.math.multiply(target_weight, 1-self.tau)
                                      for local_weight, target_weight in
                                      zip(self.Q2.get_weights(), self.target_Q2.get_weights())])

    def get_discrete_action(self, Q_value):
        if self.env.current_step is 0:  # must at least separate first stream
            return 0
        else:
            if Q_value > 0:  # separate
                action_discrete = 0
            else:  # submit
                action_discrete = 1
            return action_discrete



    def save_memory(self):
        pickle.dump(self.memory, open("./SAC/memory_data/" + str(time.time()) + ".obj", "wb"))
        pickle.dump(self.memory, open("./SAC/memory_data/memory.obj", "wb"))

    def load_memory(self):
        """
        currently just always expanding memory.obj
        """
        old_memory = pickle.load(open("./SAC/memory_data/memory.obj", "rb"))
        self.memory.buffer += old_memory.buffer

    def test_run(self):
        state = self.env.reset()
        done = False
        total_score = 0
        i = 0
        while not done:
            i +=1
            state = self.env.State.state.copy()
            action_continuous = self.Actor([state])
            Q_value = tf.minimum(self.Q1([state, action_continuous]), self.Q2([state, action_continuous]))
            if state[:, 0: self.env.n_components].max() * self.env.State.flow_norm <= self.env.min_total_flow * 1.1:
                # must submit if there is not a lot of flow, add bit of extra margin to prevent errors
                action_discrete = 1
                print("submitted due to low flow")
            else:
                if Q_value > 0:
                    action_discrete = 0  # seperate if positive reward predicted
                else:
                    action_discrete = 1
            action_continuous = np.squeeze(action_continuous, axis=0)
            action = action_continuous, action_discrete
            next_state, annual_revenue, TAC, done, info = self.env.step(action)
            reward = annual_revenue + TAC  # TAC's sign is included in the env
            total_score += reward
            print(f"step {i}: \n annual_revenue: {annual_revenue}, TAC: {TAC} \n Q_values {Q_value}, reward {reward}")
        print(total_score)