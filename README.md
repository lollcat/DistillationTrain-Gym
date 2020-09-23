# Distillation-Gym
See [paper](../Feature-double_done/Deep_Reinforcement_Learning_for_Process_Synthesis.pdf) for complete description
[GitHub](http://github.com)
<br>
Designing chemical engineering distillation processes with reinforcement learning
<br>
Using COCO simulator and ChemSep for Simulation
<br>

<br>

## Abstract
This paper demonstrates the application of reinforcement learning (RL) to process synthesis by
presenting Distillation Gym, a set of RL environments in which an RL agent is tasked with designing
a distillation train, given a user defined multi-component feed stream. Distillation Gym interfaces
with a process simulator (COCO and ChemSep) to simulate the environment. A demonstration of two
distillation problem examples are discussed in this paper (a Benzene, Toluene, P-xylene separation
problem and a hydrocarbon separation problem), in which a deep RL agent is successfully able to
learn within Distillation Gym to produce reasonable designs. Finally, this paper proposes the creation
of Chemical Engineering Gym, an all-purpose reinforcement learning software toolkit for chemical
engineering process synthesis.
 
 ## Agents
  - Soft Actor Critic (SAC)
  - Deep Deterministic Policy Gradient (DDPG)
 
## Example Problems
Distillation-Gym can be applied to any initial starting stream component selection and composition. The following examples are given:
### Benzene Toluene p-Xylene (taken from [ChemSep Examples](http://www.chemsep.org/downloads/index.html))
 ![alt text](../blob/Feature-double_done/SAC/BFDs/CONFIG%203/Attempt%202%20(best)/SAC_CONFIG_3___1598820337.9998825score_2.43.png "Benzene Toluene p-Xylene Final Design")
 ### Hydrocarbon Distillation
Agent is able to design sequence that achieves much higher revenue (through good recovery) than cost  
 ![alt text](..//blob/Feature-double_done/SAC/BFDs/CONFIG%200/best/SAC_CONFIG_0___1599080706.16091score_2.7.png "Hydrocarbon distillation")
 
