# Distillation-Gym
<br>
Currently being build
<br>
Designing chemical engineering processes with reinforcement learning
<br>
Using COCO simulator and ChemSep to simulate hydrocarbon distillation train process synthesis problem
<br>

## Agents
  - Soft Actor Critic (SAC), best performer
  - Deep Deterministic Policy Gradient (DDPG)
  
  
## TODO
  - Problem with unstable Q-values. Probably because "done" not fully complete for relevent streams at end of episode:
    - *solution*: Instead of max streams, set max column "layers" as it there can only be e.g. 5 columns in series after which both outlets get a "done". When all outlets are done, the episode is over. 
  - Not getting speedup from asynchronous run (due to ChemSep + COCO?) 
  - Once asynch problem with simulators gets fixed will need to rewrite all the asynch code as it is now all old + buggy + requires many updates
 
