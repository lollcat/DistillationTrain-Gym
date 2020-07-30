# runtime per episode (9 min / 5 episode average) is less than 2 (1.8 min) min per episode
# runtime per episode (5.5 min / 5 episode average) almost 1 min per episode
# runtime per episode (180 min for 100 episodes) - note there have been tweaks that may results in different length episodes
# 9.65 hours for 350 episodes
# tensorboard --logdir BatchMemory_Agent/logs
from Workers.Agent import Agent
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
import time

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'BatchMemory_Agent/' + 'logs/' + " with targets" + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

Agent = Agent(summary_writer=summary_writer, total_episodes=50, mem_length=3, batch_size=2)
#Agent = Agent(summary_writer=summary_writer, total_episodes=350, mem_length=1000, batch_size=64)

begin = time.time()
Agent.run_episodes()
Agent.test_run()

print(f"time is {(time.time() - begin)/60}")


def running_mean(x, N=20):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

plt.plot(Agent.history)
plt.plot(running_mean(Agent.history))
plt.show()