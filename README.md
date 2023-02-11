# Meta-Task-Generator

A small program that automatically generates simple meta-reinforcement learning
tasks from a parametrized space. The parametrization is expressive enough to
include bandit tasks, the Harlow  task, the two-step tasks, T-mazes, and other
meta-tasks.

This is a description of the Daw two-step tasks, as explained in the preprint:
![Image of the two-step meta-task from the preprint](https://github.com/ThomasMiconi/Meta-Task-Generator/blob/main/twostep.png)



It is **recommended** to first consult `simple.py`, which is simplified as much as
possible for illustrative purposes. Running the script generates and prints out
the specification for one meta-task. Conventions are included in the printout. 

The code actually used for task generation is `tasks.py`. This code contains
various tricks and workarounds to bias the generative process towards
(hopefully) more interesting or interpretable tasks.

The file `ExampleTasks.py` contains the generated specifications for the three
randomly generated meta-tasks described in the preprint, as originally produced
by the program.

Work in progress.
