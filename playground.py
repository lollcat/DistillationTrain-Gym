# tensorboard --logdir logs
from SAC.SAC_Agent.Agent import Agent
#SAC = Agent(total_eps=300, batch_size=64, max_mem_length=1000)
SAC = Agent(total_eps=200, batch_size=2)
SAC.run()
