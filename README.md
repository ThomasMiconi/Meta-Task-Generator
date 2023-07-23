# Meta-Task-Generator

A small program that automatically generates simple meta-reinforcement learning
tasks. The resulting space of accessible meta-tasks is very large and includes bandit tasks, the Harlow  task, the two-step task, T-mazes, and other
known meta-tasks from the literature.

A detailed description is available at https://arxiv.org/abs/2302.05583. 

Here is a description of the Daw two-step task in the parametrization used by this program, as explained in the preprint:
![Image of the two-step meta-task from the preprint](https://github.com/ThomasMiconi/Meta-Task-Generator/blob/main/twostep.png)



It is **recommended** to first consult `simple.py`, which is simplified as much as
possible for illustrative purposes. Running the script generates and prints out
the (textual) specification for one meta-task. Conventions are included in the printout. 

The code actually used for task generation is `tasks.py`. This code contains
various tricks and workarounds to bias the generative process towards
(hopefully) more interesting or interpretable meta-tasks.

The file `ExampleTasks.txt` contains the generated specifications for the three
randomly generated meta-tasks described in the preprint.

Work in progress.
