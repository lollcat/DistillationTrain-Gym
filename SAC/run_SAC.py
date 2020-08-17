
# tensorboard --logdir logs
import time

from SAC.SAC_Agent.Agent import Agent
SAC = Agent(total_eps=1000, batch_size=64, max_mem_length=1e4, use_load_memory=False, min_mem_length=500,
            description="fix prices")
#SAC = Agent(total_eps=2, batch_size=2)
SAC.run()
SAC.test_run()

print(f"""failed solves: {SAC.env.error_counter["error_solves"]}/{SAC.env.error_counter["total_solves"]}""")
