from SAC.Nets.Actor import GaussianPolicy as Actor
from SAC.Nets.Critic import Critic
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


log_dir = './SAC/logs/' + str(time.time())
summary_writer = tf.summary.create_file_writer(log_dir)


class Agent:
    """
    Should we be doing max(Q, 0) thing:
        (1) Not doing it would make future states "more negative" so might better learn when to stop seperating
        (2) It does seem like safe intermediate method
        (3) Better method may be to try get whether the next state ended up getting submitted or not
            (3.1) This would look like getting 2 done results per action
        (4) For not lets do the max thing
        # TODO try DDPG without max thingy
    Question: Can we use soft-Q as valid basis for whether or not to separate?
    """
    def __init__(self, env=DC_Gym(*standard_args, simple_state=True), total_eps=2e2, batch_size=64, alpha=0.5, max_mem_length=1e4, tau=0.005,
                 Q_lr=3e-4, policy_lr=3e-4, alpha_lr=3e-4, gamma=0.99, summary_writer=summary_writer, use_load_memory=False):
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
        self.alpha = tf.Variable(alpha, dtype=tf.float64)
        self.entropy_target = tf.constant(-np.prod(env.real_continuous_action_space.shape), dtype=tf.float64)

        self.Q1_optimizer = tf.keras.optimizers.Adam(Q_lr)
        self.Q2_optimizer = tf.keras.optimizers.Adam(Q_lr)
        self.Actor_optimizer = tf.keras.optimizers.Adam(policy_lr)
        self.alpha_optimizer = tf.keras.optimizers.Adam(alpha_lr)

        self.summary_writer = summary_writer
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
                action_continuous, log_pi = self.Actor.sample_action(state)
                if state[:, 0: self.env.n_components].max() * self.env.State.flow_norm <= self.env.min_total_flow * 1.1:
                    # must submit if there is not a lot of flow, add bit of extra margin to prevent errors
                    action_discrete = 1
                else:
                    Q_value = tf.minimum(self.Q1([state, action_continuous]), self.Q2([state, action_continuous]))
                    action_discrete = self.eps_greedy(Q_value, ep)
                action_continuous = np.squeeze(action_continuous, axis=0)
                action = action_continuous, action_discrete
                next_state, annual_revenue, TAC, done, info = self.env.step(action)
                tops_state, bottoms_state = next_state
                reward = annual_revenue + TAC  # TAC's sign is included in the env
                total_score += reward

                if action_discrete == 0:
                        if len(info) == 0 or (len(info) == 1 and done is True):
                            self.memory.add((state, action_continuous, reward, tops_state, bottoms_state, 1 - done))
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
                tf.summary.scalar("total score", total_score, self.steps)
                tf.summary.scalar("episode length", current_step, self.steps)
            self.total_scores.append(total_score)
        self.save_memory()


    def learn(self):
        batch = self.memory.sample(self.batch_size)
        states = np.squeeze(np.array([each[0] for each in batch]))
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch]).reshape(self.batch_size, 1)
        tops_states = np.squeeze(np.array([each[3] for each in batch]))
        bottoms_states = np.squeeze(np.array([each[4] for each in batch]))
        dones = np.array([each[5] for each in batch]).reshape(self.batch_size, 1)

        tops_actions, tops_log_pi = self.Actor.sample_action(tops_states)
        bottoms_actions, bottoms_log_pi = self.Actor.sample_action(bottoms_states)
        next_Q1 = self.target_Q1([tops_states, tops_actions]) + self.target_Q1([bottoms_states, bottoms_actions])
        next_Q2 = self.target_Q2([tops_states, tops_actions]) + self.target_Q2([bottoms_states, bottoms_actions])
        next_Q_target = tf.minimum(next_Q1, next_Q2) - self.alpha * (tops_log_pi + bottoms_log_pi)  # make target soft
        Q_expected = rewards + self.gamma * tf.maximum(next_Q_target, 0)  # cannot be negative as then would separate
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
            predicted_actions, predicted_log_pi = self.Actor.sample_action(states)
            Q1_predicted = self.Q1([states, predicted_actions])
            Q2_predicted = self.Q2([states, predicted_actions])
            Q_target = tf.minimum(Q1_predicted, Q2_predicted)
            actor_loss = tf.reduce_mean(self.alpha*predicted_log_pi - Q_target)
        actor_gradient = actor_tape.gradient(actor_loss, self.Actor.trainable_variables)
        self.Actor_optimizer.apply_gradients(zip(actor_gradient, self.Actor.trainable_variables))
        del actor_tape

        with tf.GradientTape() as alpha_tape:
            alpha_tape.watch(self.alpha)
            alpha_loss = tf.reduce_mean(self.alpha * tf.stop_gradient(-predicted_log_pi - self.entropy_target))
        alpha_gradient = alpha_tape.gradient(alpha_loss, [self.alpha])
        del alpha_tape
        self.alpha_optimizer.apply_gradients(zip(alpha_gradient, [self.alpha]))
        self.update_targets()

        with summary_writer.as_default():
            tf.summary.scalar("Q1 loss", loss_Q1, self.steps)
            tf.summary.scalar("Q2 loss", loss_Q2, self.steps)
            tf.summary.scalar("Actor loss", actor_loss, self.steps)
            tf.summary.scalar("alpha loss", alpha_loss, self.steps)
            tf.summary.scalar("alpha value", self.alpha, self.steps)


    def update_targets(self):
        self.target_Q1.set_weights([tf.math.multiply(local_weight, self.tau) +
                                    tf.math.multiply(target_weight, 1-self.tau)
                                      for local_weight, target_weight in
                                      zip(self.Q1.get_weights(), self.target_Q1.get_weights())])
        self.target_Q2.set_weights([tf.math.multiply(local_weight, self.tau) +
                                    tf.math.multiply(target_weight, 1-self.tau)
                                      for local_weight, target_weight in
                                      zip(self.Q2.get_weights(), self.target_Q2.get_weights())])

    def eps_greedy(self, Q_value, current_step, max_prob=1, min_prob=0.05):
        if self.env.current_step is 0:  # must at least separate first stream
            return 0
        else:
            explore_threshold = max(max_prob - current_step / self.eps_greedy_stop_step * (max_prob - min_prob), min_prob)
            random = np.random.rand()
            if random < explore_threshold:
                action_discrete = np.random.choice(a=[0, 1], p=[0.5, 0.5])
            else:  # exploit
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
