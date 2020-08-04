
# tensorboard --logdir SAC/logs
import time
from Utils.BFD_maker import Visualiser
from SAC.SAC_Agent.Agent import Agent
SAC = Agent(total_eps=300, batch_size=132, max_mem_length=1500)
#SAC = Agent(total_eps=200, batch_size=2)
SAC.run()
SAC.test_run()
Visualise = Visualiser(SAC.env)
G = Visualise.visualise()
G.write_png("./SAC/"+ str(time.time()) + ".png")
print(f"""failed solves: {SAC.env.error_counter["error_solves"]}/{SAC.env.error_counter["total_solves"]}""")
