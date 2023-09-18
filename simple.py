# Simpler version of the code for randomnly generating meta-reinforcement learning tasks, 
# with less structure/biasing added.

# NOTE: This code may produce sub-optimal meta-tasks (e.g.  no variation across individual 
# tasks, disconnected transition graph, redundant rules, etc.)

import numpy as np
import numpy.random as R
import json

print('''
=== Randomly  generated meta-reinforcement learning task ===
We print out the transition  matrix, the range for each state variable, the reward rules, the flag-setting rules, and the stimuli for each state.
Conventions:
-  "-1" means "empty / don't care"
- State  number 100+k (100, 101, 102...) indicates the k-th state variable (i.e. special state 0, 1, 2... respectively, each of which is replaced with a randomly chosen state number for each new task.)
- Probability value 1000+k  (1000, 1001, 1002...) indicates probability variable k  (i.e. special probability value 0, 1, 2... respectively, each of which is replaced with a randomly chosen probability in (0,1) for each new task.)
- Probability value 2000+k  (2000, 2001, 2002...) indicates "one minus probability variable k".
- Stimulus number 10000+k indicates variable stimulus k (each of which is randomly resampled for each new task)
''')

N = 4       # Number of states
NBA =  2    # Number of actions for each state
NBR = R.choice([1, 1, 1, 2, 3])     # Number of reward rules
NBFR = R.choice([1, 1, 1, 2])       # Number of flag rules


# The "simple.py" code assumes that these two values are >0:
NBSPECIALSTATES = 1
NBSPECIALPROBAS  = 2

NBVARSTIM = 2       # Number of different stimulus variables
NBFIXEDSTIM = 3     # Number of different fixed stimuli


#### From now on we generate the meta-task automatically


# Generate the range for each special state variable. When generating a new individual task, this special state variable 
# will be assigned a value randomly sampled from this range. 
specialstatesranges=[]
for ns in range(NBSPECIALSTATES):
    myrangesize = R.randint(2, N)  # A special state with range size 1 is useless. 2 to N-1 inclusive (0 cannot be a special state).            
    myrange= list(R.choice(range(1, N), size=myrangesize, replace=False))  # state 0 should not be a special state (kind of arbitrary)
    specialstatesranges.append([int(x) for x in myrange])  # The 'int' is for the JSON serialization

# Pick the stimuli for each state (more precisely, their IDs).
# Stimulus number k < 10000 indicates that a stimulus that is constant for the whole meta-task.
# Stimulus number k >= 10000 indicates a stimulus variable, to be randomly resampled for each new individual task.
# Stimulus number -1 indicates no stimulus
stims = R.choice(np.concatenate((range(NBFIXEDSTIM), 10000+np.arange(NBVARSTIM), [-1])), size=N).astype(int)


# Transition function: For each state and action, a probability distribution over all states.
T=  np.random.choice([0.0, 0.0, 0.0, 1.0, 1.0, 5.0], size=(N, NBA, N))
# Need to prevent all-0 outputs
for ns in range(N):
    for na in range(NBA):
        if np.sum(T[ns, na, :]) == 0:
            T[ns, na, np.random.randint(N)] =  1
T = T / np.sum(T, axis=2)[:, :, None]
# Next two lines optional:
T[T <.1] = 0
T = T / np.sum(T, axis=2)[:, :, None]

# Reward conditions: when do we get a reward?
# Each rule has the form: [Old state, new state, action taken, probability, value, flag]
# The rule is triggered when the conditions apply.
# Later rules override previous ones
rules = []
for nr in range(NBR):
    # Old state (state at previous time step) is always used, may be a variable (>100)
    # New state is never used, so always -1. 
    # Action may or may not be used. 
    # Value is always 1. 
    # Flag may or may not be used, if so there is a preference to require flag=1 rather than flag=0 (just a design choice)
    rule = [int(R.choice([R.randint(N), 100 + R.randint(NBSPECIALSTATES)]))  , 
            -1, 
            int(R.choice([R.randint(NBA), -1])), 
            R.choice([.2, .5, .8, 1.0, 1.0, 1000 + R.randint(NBSPECIALPROBAS), 2000 + R.randint(NBSPECIALPROBAS)]),
            1.0, 
            int(R.choice([-1, -1, 0, 1, 1]))]
    rules.append(rule)

# Flag-setting rules
# Note that flag rules may or may not be used in the reward rules
# Each rule has the form: [Old state, new state, action taken, new flag value]
# New flag value should be mostly 1 (remember that flag is assumed to be set to 0 on visiting state 0)
flagrules = []
for nr in range(NBFR):
    flagrule = [R.choice([R.randint(N), 100 + R.randint(NBSPECIALSTATES)]), -1  , R.choice([R.randint(NBA), -1]), R.choice([0,1,1,1])]
    flagrule = [int(x) for x in flagrule]
    flagrules.append(flagrule)


print("Transition table:\n", T)
for nr, r in enumerate(specialstatesranges):
    print("Range for special state", nr, ":", r)
print("Reward rules (Old state, new state, action taken, probability, value, flag):\n", rules)
print("Flag rules (Old state, new state, action taken, new flag value) (being in state 0 sets flag to 0) (may not be used!):\n", flagrules)
print("Stimulus for each state:\n", stims)


jtask = {}
jtask['T'] = T.tolist()
jtask['flagconditions'] = flagrules
jtask['rewardconditions'] = rules
jtask['stimuli'] = stims.tolist()
jtask['statevarranges'] = specialstatesranges

jtask_s = json.dumps(jtask) # , indent=2)
print("JSON version:\n", jtask_s.replace(' "', '\n"').replace('{','{\n').replace('}', '\n}'))

