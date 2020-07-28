# runtime per episode (9 min / 5 episode average) is less than 2 (1.8 min) min per episode
# runtime per episode (5.5 min / 5 episode average) almost 1 min per episode
# tensorboard --logdir logs
from Workers.Agent_target import Agent
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
import time

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'BatchMemory_Agent/' + 'logs/' + " with targets" + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

Agent = Agent(summary_writer=summary_writer, total_episodes=50, mem_length=50, batch_size=8)
#Agent = Agent(summary_writer=summary_writer, total_episodes=300, mem_length=1000, batch_size=32)

begin = time.time()
Agent.run_episodes()
print(f"time is {(time.time() - begin)/60}")


def running_mean(x, N=20):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

plt.plot(Agent.history)
plt.plot(running_mean(Agent.history))
plt.show()