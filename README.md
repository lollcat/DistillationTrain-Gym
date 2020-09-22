# Distillation-Gym

<br>
Designing chemical engineering distillation processes with reinforcement learning
<br>
Using COCO simulator and ChemSep for Simulation
<br>

## Summary
 - Given a pre-defined starting stream, product definition (required purity and selling price) and flowsheet thermophysical property specification. Design a distillation column train that maximises (revenue - total annual cost)
 - Revenue is calculated when new streams are produced according to the product definition
 - TAC is calculated by ChemSep
 - During each step the agent is given (as the (partial) observation of the environments current state) one of the process outlet stream for which it has to decide whether or not to seperate
    - If the agent decides to seperate the stream, it then has to select an operating pressure (controlled by a valve before the column) and column specification (number of stages, reflux ratio and reboil ratio). 
    - If the agent decides not to seperate the stream it becomes an outlet stream for the final process design
 
 ## Agents
  - Soft Actor Critic (SAC)
  - Deep Deterministic Policy Gradient (DDPG)
 
## Example Problems
Distillation-Gym can be applied to any initial starting stream component selection and composition. The following examples are given:
### Benzene Toluene p-Xylene (taken from [ChemSep Examples](http://www.chemsep.org/downloads/index.html))
 ![alt text](https://github.com/lollcat/DistillationTrain-Gym/blob/Feature-double_done/SAC/BFDs/CONFIG%203/Attempt%202%20(best)/SAC_CONFIG_3___1598820337.9998825score_2.43.png "Benzene Toluene p-Xylene Final Design")
 ### Hydrocarbon Distillation
Agent is able to design sequence that achieves much higher revenue (through good recovery) than cost  
 ![alt text](https://github.com/lollcat/DistillationTrain-Gym/blob/Feature-double_done/SAC/BFDs/CONFIG%200/best/SAC_CONFIG_0___1599080706.16091score_2.7.png "Hydrocarbon distillation")
 
 
## TODO
### Short Term
 - [ ] Add more complete description to this repo
 
### Long Term
  - [ ] 
