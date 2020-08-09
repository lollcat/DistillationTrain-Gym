
# tensorboard --logdir logs
import time
from Utils.BFD_maker import Visualiser
from Hard_Actor_Critic.HAC_Agent.Agent import Agent
HAC = Agent(total_eps=500, batch_size=138, max_mem_length=1000)
#SAC = Agent(total_eps=2, batch_size=2)
HAC.run()
HAC.test_run()
Visualise = Visualiser(HAC.env)
G = Visualise.visualise()
G.write_png("./SAC/" + str(time.time()) + ".png")
print(f"""failed solves: {HAC.env.error_counter["error_solves"]}/{HAC.env.error_counter["total_solves"]}""")
