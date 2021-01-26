# Distillation-Gym
 - See [paper](../Feature-double_done/Deep_Reinforcement_Learning_for_Process_Synthesis.pdf) for complete description
 - Designing chemical engineering distillation processes with reinforcement learning
 - Using [COCO simulator](https://www.cocosimulator.org/) and [ChemSep](http://www.chemsep.org/program/index.html) for Simulation
 - To run the agent, try [this file](../Feature-double_done/SAC/run_SAC.py) and see details in "About this repo" at the bottom of this page

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
  - Soft Actor Critic (SAC) - this is the main agent that was used
  - Deep Deterministic Policy Gradient (DDPG)
 
## Example Problems
Distillation-Gym can be applied to any initial starting stream component selection and composition. The following examples are given:
### Benzene Toluene p-Xylene (taken from [ChemSep Examples](http://www.chemsep.org/downloads/index.html))
 ![alt text](../Feature-double_done/SAC/BFDs/CONFIG%203/Attempt%202%20(best)/SAC_CONFIG_3___1598820337.9998825score_2.43.png "Benzene Toluene p-Xylene Final Design")
 ### Hydrocarbon Distillation
Agent is able to design sequence that achieves much higher revenue (through good recovery) than cost  
 ![alt text](../Feature-double_done/SAC/BFDs/CONFIG%200/best/SAC_CONFIG_0___1599080706.16091score_2.7.png "Hydrocarbon distillation")
 
 
 ## About this Repo
 The important folders are:
  - Env: Contains the RL environment (where python and COCO simulator speak using the COM interface)
  - SAC: The main agent
  - Utils: Some useful extra stuff (e.g. BFD generator)\

The other folders and files are just playing around with some other stuff (e.g. DDPG agent)\
See [requirements.txt](../Feature-double_done/requirements.txt) for required packages\
Both COCO and ChemSep need to be installed: To install these go to https://www.cocosimulator.org/index_download.html
 
To cite please use:

@article{midgley2020deep,
  title={Deep Reinforcement Learning for Process Synthesis},
  author={Midgley, Laurence Illing},
  journal={arXiv preprint arXiv:2009.13265},
  year={2020}
}
