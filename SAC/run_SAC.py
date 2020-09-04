
# tensorboard --logdir logs
from SAC.SAC_Agent.Agent import Agent
SAC = Agent(total_eps=2000, batch_size=256, max_mem_length=1e4, use_load_memory=True,
            description="big_batch", COCO_flowsheet_number=2, extra_explore_noise=True)
#SAC = Agent(total_eps=2, batch_size=2)
SAC.run()
SAC.test_run()

print(f"""failed solves: {SAC.env.error_counter["error_solves"]}/{SAC.env.error_counter["total_solves"]}""")
