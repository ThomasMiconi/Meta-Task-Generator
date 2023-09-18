import numpy as np
import json

print('''
=== Randomly  generated meta-reinforcement learning task ===
We print out the transition  matrix, the range for each state variable, the reward rules, the flag-setting rules, and the stimuli for each state.
We also print out a JSON representation of the meta-task.
Conventions:
-  "-1" means "empty / don't care"
- State  number 100+k (100, 101, 102...) indicates the k-th special-state variable (i.e. special state 0, 1, 2... respectively, each of which is replaced with a randomly chosen state number for each new task.)
- Probability value 1000+k  (1000, 1001, 1002...) indicates probability variable k  (i.e. special probability value 0, 1, 2... respectively, each of which is replaced with a randomly chosen probability in (0,1) for each new task.)
- Probability value 2000+k  (2000, 2001, 2002...) indicates "one minus probability variable k".
- Stimulus number 10000+k indicates variable stimulus k (each of which is randomly resampled for each new task)
''')

### This entire code generates one single meta-learning task.

# More precisely, after setting the paraameters, we repeatedly try to generate
# a meta-task and test whether it meets certain criteria. If so, we accept it,
# otherwise we try again.

# Most of the complexity  in this code results from attempts at introducing
# some structure and some filtering in the process, in the hope of biasing it
# towards more interesting / interpretable meta-tasks.


### The following parameters are common to all meta-learning tasks that this code generates:

N = 4       # Number of states
NBA =  2    # Number of actions for each state
PROBAUSEOLDSTATE=  1.0      # Probability that a rule condition will include the starting state
PROBAUSENEWSTATE=  0.0      # Probability that a rule condition will include the new state
PROBAUSEACTION=  .33        # Probability that a rule condition will include the action
NBSTATEVARIABLES= 1         # Number of different state variables
NBSPECIALPROBAS = 2         # Number of different probability variables
PROBANOSTIM =  .2           # Probability that a given state provides no stimulus/observation


NBVARSTIM = 2       # Number of different stimulus variables
NBFIXEDSTIM = 3     # Number of different fixed stimuli



# We will keep generating meta-tasks until we find one that meets certain criteria, at which point we exit


OK = False # Should we accept this generated meta-task?
while not OK:

    ### The following parameters are randomly chosen for each meta-task generation:

    NBR = np.random.choice([1, 1, 1, 2, 3])     # Number of reward rules
    NBFR = np.random.choice([1, 1, 1, 2])       # Number of flag rules
    
    # Parameters for generating rules (for rewards and flags):
    # Note that many of these will be zero for most generated meta-tasks. This is a design choice - we assume most meta-tasks do not need the relevant variability. 
    PROBASTATEISVARIABLE = np.random.choice([0.0, 0.0, .5])  # Probability that the 'state' component of a given rule will actually be a variable. 
    PROBAREWARDISPROBABILISTIC = np.random.choice([0.0, 0.0, 0.0, 1.0]) # Probaility that the reward for a given rule is probabilistic (either no reward is probabilistic, or all are)
    PROBAREWARDPROBAISVARIABLE =  np.random.choice([.2, .5, .8]) # *If* a reward is probabilisitc, probability that it's a variable
    PROBAEACHRULEUSESFLAG = np.random.choice([0.0, 0.0, .5])  # Flags should be used sparingly



#### From now on  we generate the meta-task automatically

    # Generate the range for each special state variable. When generating a new individual task, this special state variable 
    # will be assigned a value randomly sampled from this range. 
    specialstatesranges=[]
    for ns in range(NBSTATEVARIABLES):
        myrangesize = 2 if np.random.rand() < .5 else np.random.randint(2, N)  # A special state with range size 1 is useless. 2 to N-1 inclusive (0 cannot be a special state).            
        myrange= list(np.random.choice(range(1, N), size=myrangesize, replace=False))  # state 0 should not be a special state (kind of arbitrary)
        specialstatesranges.append([int(x) for x in myrange])

    # Pick the stimuli (more precisely, their IDs).
    # Stimulus number k < 1000 indicates that a stimulus that is constant for the whole meta-task.
    # Stimulus number k >= 1000 indicates a stimulus variable, to be randomly resampled for each new individual task.
    stims = np.zeros(N).astype(int)
    nbdiffvarstims = 1
    while nbdiffvarstims == 1:
        for ns in range(N):
            stims[ns] = np.random.randint(NBFIXEDSTIM) if np.random.rand()  < .666 else 10000 + np.random.randint(NBVARSTIM) 
            if np.random.rand() < PROBANOSTIM:
                stims[ns] = -1
        nbdiffvarstims = len(np.unique(stims[stims >= 10000])) # 1 task-variable stimulus doesn't induce real variation over tasks, because it ends up being "the state that's  not fixed/nothing".  Must have at least 2  different vari stims, if any.

    # Transition function: For each state and action, a probability distribution over all states.
    # In many case, this distribution should be a one-hot vector. Even most of the remaining cases should be two-hot vectors(only two possilbe options).
    # But not always, sometimes more options.
    # Furthermore, the same distribution (or sometimes, if two-hot, its mirror image) should  sometimes be replicated over all actions.
    # Only allowed probabilities are binary (equal prob amobg non-zero), or one bigger than the others.
    # The actual process, and the probability values used below, are somewhat arbitrary. They seem to produce OK results.
    T=  np.zeros((N, NBA, N))
    for ns in range(N):
        for na in range(NBA):
            nextS  = np.random.randint(N)
            T[ns, na,  nextS]  = 1
            if np.random.rand() < .33 :
                T[ns, na,  nextS]  *= 5  # If  there are other non-zero probabilities, this one will be the highest
            if np.random.rand()  < .5:
                T[ns, na,  np.random.randint(N)]  = 1  # Yes, might be the same
                if np.random.rand()  < .33:
                    T[ns, na,  np.random.randint(N)]  = 1  
                    if np.random.rand()  < .33:
                        T[ns, na,  np.random.randint(N)]  = 1 
        if np.random.rand() < .5:   # Make all actions have the same output distributions, possibly flipped
            for na2 in range(1, NBA):
                T[ns, na2, :] = T[ns, 0, :]
            nzp = np.nonzero(T[ns, 0, :])[0]
            if len(nzp) > 1  and np.random.randn()  < .5:  # Flip two outcomes for different actions? (Only if >1 nonzero values - may amount to noop if all nonzero values are 1, that's fine)
                nanotflipped = np.random.randint(NBA)
                (p1, p2) = np.random.choice(nzp, size=2, replace=False)
                #print("Flipping state", ns, "actions outcome at positions", p1, "and", p2, " for all actions other than action", nanotflipped)
                for na2 in range(0, NBA):
                    if na2  == nanotflipped:
                        continue
                    tmp = T[ns, na2, p1] 
                    T[ns, na2, p1] = T[ns, na2, p2]
                    T[ns, na2, p2]  = tmp

    T = T / np.sum(T, axis=2)[:, :, None]

    # Reward rules
    # Old state, new state, action taken, probability, value, flag
    # Later rules override previous ones
    rules = []
    for nr in range(NBR):
        rule = [-1, -1, -1, 0, 0, -1]
        while (rule[0] ==  -1  and rule[1] == -1 and rule[2] == -1) or (rule[0]  ==  1 and rule[5]  == 1):  # Rules in state 0 requiring flag 1 will never apply
            if np.random.rand() < PROBAUSEOLDSTATE:
                rule[0] =  int(np.random.randint(N))
            if np.random.rand() < PROBAUSENEWSTATE:
                rule[1] =  int(np.random.randint(N))
            if np.random.rand() < PROBAUSEACTION:
                rule[2] =  int(np.random.randint(NBA))
            rule[3] = 1.0 # Probability is 1.0 for now but may be modified below
            rule[4] = 1.0 # Value always 1.0
            
            # Should some of the rules make use of the state variables? (The precise identity of which will be picked when we generate an actual new instance/task)
            if rule[0] != -1 and np.random.rand() < PROBASTATEISVARIABLE:
                rule[0] = 100 + np.random.randint(NBSTATEVARIABLES)
            if rule[1] != -1 and np.random.rand() < PROBASTATEISVARIABLE:
                rule[1] = 100 + np.random.randint(NBSTATEVARIABLES)
            # Should some of the rules make use of the flag?
            if np.random.rand() < PROBAEACHRULEUSESFLAG:
                rule[5] =  np.random.choice([1.0, 1.0, 1.0, 0.0]) # Mostly look for set flag (just a design choice)
            # Probabiliistic rewards? (and possibly probability variables?)
            if np.random.rand() < PROBAREWARDISPROBABILISTIC:
                rule[3] = np.random.choice([.2, .5, .8, 1.0])
                if np.random.rand() < PROBAREWARDPROBAISVARIABLE:    # Use a probability variable. Notice the indent !
                    if np.random.rand() < .5:
                        rule[3] = 1000 * (1+np.random.randint(2)) + np.random.randint(NBSPECIALPROBAS) # i.e. variable
                    else:  
                        # We use "1 minus" the k-th probability variable
                        rule[3] = 2000 * (1+np.random.randint(2)) + np.random.randint(NBSPECIALPROBAS) # i.e. variable

        rules.append(rule)


    # Flag-setting rules (in addition to the standard rule  that transitioning to state 0 sets flag to 0)
    # Old state, new state, action taken, new flag value 
    # New flag value should be mostly 1
    flagrules = []
    for nr in range(NBFR):
        ok0 = False
        while not ok0: 
            flagrule = [-1, -1, -1, 1.0]
            while flagrule[0] ==  -1  and flagrule[1] == -1 and flagrule[2] == -1:
                if np.random.rand() < PROBAUSEOLDSTATE:
                    flagrule[0] =  np.random.randint(N)
                if np.random.rand() < PROBAUSENEWSTATE:
                    flagrule[1] =  np.random.randint(N)
                if np.random.rand() < PROBAUSEACTION:
                    flagrule[2] =  np.random.randint(NBA)
            flagrule[3] = np.random.choice([1, 1, 1, 0])
            ok0 = flagrule[0] != 0 or flagrule[1] != -1 or flagrule[2] != -1 # State 0 cannot unconditionally set the flag
        flagrules.append([int(x) for x in flagrule])
    # Should some of the flag rules make use of the special states? (The precise identity of which will be picked when we generate an actual new instance/task)
    for nr in range(NBFR):
        if flagrules[nr][0] != -1 and np.random.rand() < PROBASTATEISVARIABLE:
            flagrules[nr][0] = 100 + np.random.randint(NBSTATEVARIABLES)
        if flagrules[nr][1] != -1 and np.random.rand() < PROBASTATEISVARIABLE:
            flagrules[nr][1] = 100 + np.random.randint(NBSTATEVARIABLES)
    flagrules[0][3] = 1  # There should be at least one rule that actually sets the flag  (note  that flag rules may  not be  used)



    # Now we test whether the generated meta-task fulfilles some criteria for acceptance:

    # At least two states must have different outcomes for either action
    somediffoutcomes = 0
    for ns in range(N):
        if np.any(T[ns,0] != T[ns, 1]):
            somediffoutcomes += 1
    # *Something* must be variable across instances of the meta-task (note that random rewarrd probabilities only induce  true variation if there's more than one  reward rule):
    somevar  = np.any(stims>99) or np.any(np.logical_and(np.array(rules)>99, np.array(rules)<1000)) or  (np.any(np.array(rules)>999) and NBR > 1)
    # There must be some way  out of 0 -  0 must not be a  terminal state:
    wayoutof0 = np.any(T[0, :, 0]  < 1.0)
    # Every state must be reachable:
    someunreachable =0
    for ns in range(N):
        sumprobastons = 0
        for ns2 in range(N):
            if ns2  == ns:
                continue
            sumprobastons += np.sum(T[ns2,  :, ns])
        if sumprobastons  == 0:
            someunreachable = 1
            break

    OK = somevar and somediffoutcomes > 1 and  wayoutof0 and not someunreachable

    # Might also want to add that the graph must be weakly
    # connected (no completely separate sub-graphs)

# OK = True
print("PROBASTATEISVARIABLE:", PROBASTATEISVARIABLE, "PROBAREWARDISPROBABILISTIC:", 
                PROBAREWARDISPROBABILISTIC, "PROBAREWARDPROBAISVARIABLE:", PROBAREWARDPROBAISVARIABLE, 
                "PROBAEACHRULEUSESFLAG:", PROBAEACHRULEUSESFLAG)   # Yeah, should use dict...

print("Transition table:\n", T)
for nr, r in enumerate(specialstatesranges):
    print("Range for special state", nr, ":", r)
print("Reward conditions (Old state, new state, action taken, probability, value, flag):\n", rules)
print("Flag-setting conditions (Old state, new state, action taken, new flag value) (being in state 0 sets flag to 0) (the flag may not be used! check reward rules):\n", flagrules)
print("Stimulus for each state:\n", stims)

# En informatique, tout finit toujours par du JSON
jtask = {}
jtask['T'] = T.tolist()
jtask['flagconditions'] = flagrules
jtask['rewardconditions'] = rules
jtask['stimuli'] = stims.tolist()
jtask['statevarranges'] = specialstatesranges

jtask_s = json.dumps(jtask) # , indent=2)
print("JSON version:\n", jtask_s.replace(' "', '\n"').replace('{','{\n').replace('}', '\n}'))

