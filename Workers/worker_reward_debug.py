def debug_class(worker_no):
    if worker_no == 0:
        from Workers.worker import Worker as Worker_original
    else:
        from Workers.worker_reward import Worker_reward as Worker_original
    class Worker(Worker_original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def run(self):
            self.env = self.env(*self.env_args)
            self.state = self.env.reset()
            for _ in range(100):
                # Collect some experience
                experience = self.run_n_steps()
                # Update the global networks using local gradients
                self.update_global_parameters(experience)
                # Stop once the max number of global steps has been reached
                if self.max_global_steps is not None and self.global_step >= self.max_global_steps:
                    return f'worker {self.name}, step: {self.global_step}'
    return Worker