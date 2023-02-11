# Meta-Task-Generator

A small program that automatically generates simple meta-reinforcement learning
tasks from a parametrized space. The parametrization is expressive enough to
include bandit tasks, the Harlow  task, the two-step tasks, T-mazes, and other
meta-tasks.

This is a description of the Daw two-step tasks, as explained in the preprint:
![Image of the two-step meta-task from the preprint](https://github.com/ThomasMiconi/Meta-Task-Generator/blob/main/twostep.png)



The main code is in `tasks.py`. Running the script generates and prints out the
specification for one meta-task. Conventions are included in the printout. 

The file `ExampleTasks.py` contains the generated specifications for the three
randomly generated meta-tasks described in the preprint, as originally produced
by the program.

Work in progress.
