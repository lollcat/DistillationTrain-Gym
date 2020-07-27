"""PRIMARY CONFIG"""
agent_config = 0
max_global_steps = 6*150


import tensorflow as tf
if agent_config == 0:
    CONFIG_string = "Simple State, Simple reward"
    from Env.DC_gym import DC_Gym
    from Nets.Critic_simple import Critic
    from Workers.worker import Worker
else:
    CONFIG_string = "Simple State, Split reward"
    from Env.DC_gym_reward import DC_gym_reward as DC_Gym
    from Nets.Critic import Critic
    from Workers.worker_reward import Worker_reward as Worker
from Env.STANDARD_CONFIG import CONFIG
from Nets.P_actor import ParameterAgent

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

"""SECONDARY CONFIG"""
standard_args = CONFIG(1).get_config()
alpha = 0.0001
beta = alpha*10
steps_per_update = 5
env = DC_Gym(*standard_args)
n_continuous_actions = env.continuous_action_space.shape[0]
n_discrete_actions = env.discrete_action_space.n
state_shape = env.observation_space.shape


num_workers = multiprocessing.cpu_count()
global_counter = itertools.count()
returns_list = []

param_model, param_optimizer = ParameterAgent(beta, n_continuous_actions, state_shape).build_network()
dqn_model, dqn_optimizer = Critic(alpha, n_continuous_actions, state_shape,).build_network()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time + CONFIG_string
summary_writer = tf.summary.create_file_writer(log_dir)

start_time = time.time()
workers = []
for worker_id in range(num_workers):
    worker = Worker(
        name=worker_id,
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
    workers.append(worker)

coord = tf.train.Coordinator()
worker_fn = lambda worker_: worker_.run(coord)
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(worker_fn, workers, timeout=40)
"""
"""
print(f"time taken is {(time.time() - start_time)/60} minutes")
def running_mean(x, N=20):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

plt.plot(returns_list)
plt.plot(running_mean(returns_list))
#plt.yscale('log')
plt.show()
