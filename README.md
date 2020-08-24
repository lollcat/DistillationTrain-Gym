# Distillation-Gym
<br>
Currently being build
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
 - To simplfy the state a tree structure Markov Decision Process was created (*Explain further*)
 
## Example Problems
Distillation-Gym can be applied to any initial starting stream component selection and composition. The following examples are given:
### Benzene Toluene p-Xylene (taken from [ChemSep Examples](http://www.chemsep.org/downloads/index.html))
Interstingly the current best solution uses 3 columms instead of the expected 2 columns 
 ![alt text](https://github.com/lollcat/DistillationTrain-Gym/blob/Feature-double_done/SAC/BFDs/CONFIG3/SAC_CONFIG_3___high_alpha_lr1598053716.1167474score_2.42.png "Benzene Toluene p-Xylene Final Design")
 
 
## Agents
  - Soft Actor Critic (SAC)
  - Deep Deterministic Policy Gradient (DDPG)
    
## TODO
### Short Term
  - [ ] Write out this README fully
  - [ ] Create example solution in COCO
    - Actually quite difficult to do
    - If we do Luyben example then the paper can act as an example solution
  - [ ] Run with varying starting streams
    - New Luyben ChemSep hydrocarbon problem looks good
    - Even one of the ChemSep examples with 3 streams could be good as simple example
  - [ ] Change action variables (i.e. not reflux ratio etc)
    - but other variables aren't ratios so seem like they don't necessarily "make sense" as action variables (e.g. condensor duty would be weird for the agent to choose)
    - selecting component splits would be cool but seems like just gives solve errors with ThomsonKing example
    - maybe ask for advice here

### Long Term
  - Not getting speedup from asynchronous run (due to ChemSep + COCO?) 
  - Once asynch problem with simulators gets fixed will need to rewrite all the asynch code as it is now all old + buggy + requires many updates
 
 
