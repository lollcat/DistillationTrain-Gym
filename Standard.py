from Workers.Agent import Agent
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + "env2 agent" + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

Agent = Agent(summary_writer=summary_writer, total_episodes=100, mem_length=50, batch_size=16)

Agent.populate_memory()
print("memory polulated")
Agent.run_episodes()

def running_mean(x, N=20):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

plt.plot(Agent.history)
plt.plot(running_mean(Agent.history))
plt.show()