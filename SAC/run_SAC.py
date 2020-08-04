
# tensorboard --logdir SAC/logs
import time
from Utils.BFD_maker import Visualiser
from SAC.SAC_Agent.Agent import Agent
SAC = Agent(total_eps=600, batch_size=256, max_mem_length=2000)
#SAC = Agent(total_eps=2, batch_size=2)
SAC.run()
SAC.test_run()
Visualise = Visualiser(SAC.env)
G = Visualise.visualise()
G.write_png("./SAC/"+ str(time.time()) + ".png")
print(f"""failed solves: {SAC.env.error_counter["error_solves"]}/{SAC.env.error_counter["total_solves"]}""")
