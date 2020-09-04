"""
If asynchronous run is getting speedup then average solve time should remain constant
Because more workers are running, the total time for the program to run should go down
This program shows the solve time going up as more workers are added - preventing speedup from asynchronous running
note: I only ran for a few steps so there may be high variance in these average solve time estimates
I get the following average solve times:
    n_workers = 1, average solve_time = 10 seconds
    n_workers = 2, average solve_time = 19 seconds
    n_workers = 4, average solve_time =  42 seconds
"""
from Worker import Worker
import itertools
import concurrent.futures
import multiprocessing
import os

total_steps = 12
n_workers = 4  # multiprocessing.cpu_count() # choose number of workers here

global_counter = itertools.count()
solve_time_list = []


workers = [Worker(global_counter, solve_time_list, total_steps,
                  COCO_doc_path=os.path.join(os.getcwd(), "LuybenExamplePart.fsd")) for _ in range(n_workers)]
run_worker = lambda worker_: worker_.run()

with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
    executor.map(run_worker, workers, timeout=60)

print(f"Average solve time for {n_workers} workers is {sum(solve_time_list)/len(solve_time_list)} seconds")