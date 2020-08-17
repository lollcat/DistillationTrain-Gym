# Distillation-Gym
<br>
Currently being build
<br>
Designing chemical engineering processes with reinforcement learning
<br>
Using COCO simulator and ChemSep to simulate hydrocarbon distillation train process synthesis problem
<br>

## Agents
  - Soft Actor Critic (SAC)
  - Deep Deterministic Policy Gradient (DDPG)
    
## TODO
### Short Term
  - [ ] Create example solution in COCO
    - Actually quite difficult to do
  - [ ] Run with varying starting streams
  - [ ] Change action variables (i.e. not reflux ratio etc)
    - but other variables aren't ratios so seem like they don't necessarily "make sense" as action variables
    - maybe ask for advice here

### Long Term
  - Not getting speedup from asynchronous run (due to ChemSep + COCO?) 
  - Once asynch problem with simulators gets fixed will need to rewrite all the asynch code as it is now all old + buggy + requires many updates
 
 
