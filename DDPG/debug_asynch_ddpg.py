DEBUG = 1
if DEBUG == 0:
    from Env.DC_gym import DC_Gym
    from DDPG.Nets.Critic_simple import Critic
else:
    from Env.DC_gym_reward import DC_gym_reward as DC_Gym
    from DDPG.Nets.Critic import Critic
import tensorflow as tf
from DDPG.Workers.worker_reward_debug import debug_class
Worker = debug_class(DEBUG)
from Env.STANDARD_CONFIG import CONFIG

from DDPG.Nets.P_actor import ParameterAgent

import matplotlib.pyplot as plt

import datetime
import time
import numpy as np
import multiprocessing
import concurrent.futures
import itertools

# tensorboard --logdir logs

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""CONFIG"""
CONFIG_string = "worker split reward, simplified State"
standard_args = CONFIG(1).get_config()
max_global_steps = 6*400
alpha = 0.0001
beta = alpha*10
steps_per_update = 5
env = DC_Gym(*standard_args)
n_continuous_actions = env.continuous_action_space.shape[0]
n_discrete_actions = env.discrete_action_space.n
state_shape = env.observation_space.shape


global_counter = itertools.count()
returns_list = []

param_model, param_optimizer = ParameterAgent(beta, n_continuous_actions, state_shape).build_network()
dqn_model, dqn_optimizer = Critic(alpha, n_continuous_actions, state_shape,).build_network()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time + CONFIG_string
summary_writer = tf.summary.create_file_writer(log_dir)



worker = Worker(
    name="test",
    global_network_P=param_model,
    global_network_dqn=dqn_model,
    global_optimizer_P=param_optimizer,
    global_optimizer_dqn=dqn_optimizer,
    global_counter=global_counter,
    env=DC_Gym,
    env_args=standard_args,
    n_continuous_actions=n_continuous_actions,
    max_global_steps=max_global_steps,
    returns_list=returns_list,
    n_steps=steps_per_update,
    summary_writer=summary_writer)

worker.run()
