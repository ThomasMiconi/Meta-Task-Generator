import numpy as np


print('''
=== Randomly  generated meta-reinforcement learning task ===
We print out the transition  matrix, the range for each state variable, the reward rules, the flag-setting rules, and the stimuli for each state.
Conventions:
-  "-1" means "empty / don't care"
- State  number 100+k (100, 101, 102...) indicates the k-th special state (i.e. special state 0, 1, 2... respectively, each of which is replaced with a randomly chosen state number for each new task.)
- Probability value 1000+k  (1000, 1001, 1002...) indicates special probability value k  (i.e. special probability value 0, 1, 2... respectively, each of which is replaced with a randomly chosen probability in (0,1) for each new task.)
- Probability value 2000+k  (2000, 2001, 2002...) indicates "1 - special probability value k".
- Stimulus number 1000+k indicates varialbe stimulus k (each of which is randomly resampled for each new task)
''')

N = 4
NBA =  2
NBR = np.random.choice([1, 1, 1, 2, 3])
NBFR = np.random.choice([1, 1, 1, 2])
PROBAUSEOLDSTATE=  1.0
PROBAUSENEWSTATE=  0.0
PROBAUSEACTION=  .33
NBSPECIALSTATES= 1  
NBSPECIALPROBAS = 2
STIMSIZE= 5
PROBANOSTIM =  .2

#### Now, set variables for the new meta-task:

PROBAUSESPECIALSTATE = np.random.choice([0.0, 0.0, .5])  # Most meta-tasks don't need special states
#PROBAUSEPROBABILISTICREWARDS = np.random.choice([0.0, 0.0, 0.0, .2, .5, .8]) # Or probabilistic rewards either
PROBAUSEPROBABILISTICREWARDS = np.random.choice([0.0, 0.0, 0.0, 1.0]) # Or probabilistic rewards either (but  if they do, all rewards should be probabilistic)
PROBAUSEVARREWARDPROB =  np.random.choice([.2, .5, .8]) # *If* a reward is probabilisitc, proba that it's also variable across instances/tasks 




PROBAUSEFLAG = np.random.choice([0.0, 0.0, .5])  # State variables / "flags" should be used sparingly



NBVARSTIM = 2
NBFIXEDSTIM = 3

#### From now on  we generate the meta-task automatically

OK = False
while not OK:
    specialstatesranges=[]
    for ns in range(NBSPECIALSTATES):
        myrangesize = 2 if np.random.rand() < .5 else np.random.randint(2, N)  # A special state with range size 1 makes no sense.  2 to N-1 inclusive (excludes 0).            
        myrange= list(np.random.choice(range(1, N), size=myrangesize, replace=False))  # state 0 should not be a special state (kind of arbitrary)
        specialstatesranges.append(myrange)

    fixedstims = []
    for ns in range(NBFIXEDSTIM):
        fixedstims.append(np.random.randint(2, size=STIMSIZE))

    stims = np.zeros(N).astype(int)
    nbdiffvarstims = 1
    while nbdiffvarstims == 1:
        for ns in range(N):
            stims[ns] = np.random.randint(NBFIXEDSTIM) if np.random.rand()  < .666 else 1000 + np.random.randint(NBVARSTIM) 
            if np.random.rand() < PROBANOSTIM:
                stims[ns] = -1
        nbdiffvarstims = len(np.unique(stims[stims >= 1000])) # 1 task-variable stimulus doesn't induce real variation over tasks, because it ends up being "the state that's  not fixed/nothing".  Must have at least 2  different vari stims, if any.

    # Transition function: For each state and action, a probability distribution over all states.
    # In many case, this distribution should be a one-hot vector. Even most of the remaining cases should be two-hot vectors(only two possilbe options).
    # But not always, sometimes more options (though probably never all possible states)
    # Furthermore, the same distribution (or sometimes, if two-hot, its mirror image) should  often be replicated over all actions.
    # Only allowed probabilities are binary (equal prob amobg non-zero), or one bigger than the others.
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
            if len(nzp) > 1  and np.random.randn()  < .5:  # Flip? (Only if >1 nonzero values - may amount to noop if all nonzero values are 1, that's fine)
                nanotflipped = np.random.randint(NBA)
                (p1, p2) = np.random.choice(nzp, size=2, replace=False)
                #print("Flipping state", ns, "actions outcome at positions", p1, "and", p2, "other than action", nanotflipped)
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
                rule[0] =  np.random.randint(N)
            if np.random.rand() < PROBAUSENEWSTATE:
                rule[1] =  np.random.randint(N)
            if np.random.rand() < PROBAUSEACTION:
                rule[2] =  np.random.randint(NBA)
            rule[3] = 1.0 # np.random.choice([.2, .8, 1.0, 1.0])
            rule[4] = 1.0
        rules.append(rule)

    # Should some of the rules make use of the special states? (The precise identity of which will be picked when we generate an actual new instance/task)
    for nr in range(NBR):
        if rules[nr][0] != -1 and np.random.rand() < PROBAUSESPECIALSTATE:
            rules[nr][0] = 100 + np.random.randint(NBSPECIALSTATES)
        if rules[nr][1] != -1 and np.random.rand() < PROBAUSESPECIALSTATE:
            rules[nr][1] = 100 + np.random.randint(NBSPECIALSTATES)

    # Should some of the rules make use of the flag?
    for nr in range(NBR):
        if np.random.rand() < PROBAUSEFLAG:
            rules[nr][5] =  np.random.choice([1.0, 1.0, 1.0, 0.0]) # Mostly look for set flag (just a design choice)


    # Probabiliistic rewards?
    for nr in range(NBR):
        if np.random.rand() < PROBAUSEPROBABILISTICREWARDS:
            rules[nr][3] = np.random.choice([.2, .5, .8, 1.0])
            if np.random.rand() < PROBAUSEVARREWARDPROB:    # Notice the indent !
                rules[nr][3] = 1000 * (1+np.random.randint(2)) + np.random.randint(NBSPECIALPROBAS)   # i.e. variable  


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
        flagrules.append(flagrule)
    # Should some of the flag rules make use of the special states? (The precise identity of which will be picked when we generate an actual new instance/task)
    for nr in range(NBFR):
        if flagrules[nr][0] != -1 and np.random.rand() < PROBAUSESPECIALSTATE:
            flagrules[nr][0] = 100 + np.random.randint(NBSPECIALSTATES)
        if flagrules[nr][1] != -1 and np.random.rand() < PROBAUSESPECIALSTATE:
            flagrules[nr][1] = 100 + np.random.randint(NBSPECIALSTATES)
    flagrules[0][3] = 1  # There should be at least one rule that actually sets the flag  (note  that flag rules may  not be  used)

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
print("PROBAUSESPECIALSTATE:", PROBAUSESPECIALSTATE, "PROBAUSEPROBABILISTICREWARDS:", 
                PROBAUSEPROBABILISTICREWARDS, "PROBAUSEVARREWARDPROB:", PROBAUSEVARREWARDPROB, 
                "PROBAUSEFLAG:", PROBAUSEFLAG)   # Yeah, should use dict...

print("Transition table:\n", T)
for nr, r in enumerate(specialstatesranges):
    print("Range for special state", nr, ":", r)
print("Reward rules (Old state, new state, action taken, probability, value, flag):\n", rules)
print("Flag rules (Old state, new state, action taken, new flag value) (being in state 0 sets flag to 0) (may not be used!):\n", flagrules)
print("Stimulus for  each state:\n", stims)
