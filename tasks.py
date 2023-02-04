import numpy as np

N = 3
NBA =  2
NBR = 3
PROBAUSEOLDSTATE=  1.0
PROBAUSENEWSTATE=  0.0
PROBAUSEACTION=  .5
NBSPECIALSTATES= 1  
STIMSIZE= 5
PROBANOSTIM =  .2

#### Now, set variables for the new meta-task:

PROBAUSESPECIALSTATE = np.random.choice([0.0, 0.0, .5])  # Most meta-tasks don't need special states
#PROBAUSEPROBABILISTICREWARDS = np.random.choice([0.0, 0.0, 0.0, .2, .5, .8]) # Or probabilistic rewards either
PROBAUSEPROBABILISTICREWARDS = np.random.choice([0.0, 0.0, 0.0, 1.0]) # Or probabilistic rewards either (but  if they do, all rewards should be probabilistic)
PROBAUSEVARREWARDPROB =  np.random.choice([.2, .5, .8]) # *If* a reward is probabilisitc, proba that it's also variable across instances/tasks 
NBVARSTIM = 2
NBFIXEDSTIM = 2

#### From now on  we generate the meta-task automatically

OK = False
while not OK:
    specialstatesranges=[]
    for ns in range(NBSPECIALSTATES):
        myrangesize = np.random.randint(2, N+1)  # A special state with range size 1 makes no sense.  2 to N inclusive.
        myrange= list(np.random.choice(range(N), size=myrangesize, replace=False))
        specialstatesranges.append(myrange)

    fixedstims = []
    for ns in range(NBFIXEDSTIM):
        fixedstims.append(np.random.randint(2, size=STIMSIZE))

    stims = np.zeros(N).astype(int)
    nbdiffvarstims = 1
    while nbdiffvarstims == 1:
        for ns in range(N):
            stims[ns] = np.random.randint(NBFIXEDSTIM) if np.random.rand()  < .5 else 1000 + np.random.randint(NBVARSTIM) 
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
            if np.random.rand() < .5:
                T[ns, na,  nextS]  *= 5  # If  there are other non-zero probabilities, this one will be the highest
            if np.random.rand()  < .5:
                T[ns, na,  np.random.randint(N)]  = 1  # Yes, might be the same
                if np.random.rand()  < .5:
                    T[ns, na,  np.random.randint(N)]  = 1  
                    if np.random.rand()  < .5:
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
    # Old state, new state, action taken, probability, value
    # Later rules override previous ones
    rules = []
    for nr in range(NBR):
        rule = [-1, -1, -1, 0, 0]
        while rule[0] ==  -1  and rule[1] == -1 and rule[2] == -1:
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

    for nr in range(NBR):
        if np.random.rand() < PROBAUSEPROBABILISTICREWARDS:
            rules[nr][3] = np.random.choice([.2, .5, .8, 1.0])
            if np.random.rand() < PROBAUSEVARREWARDPROB:    # Notice the indent !
                rules[nr][3] = 1000   # i.e. "choose it  at instace/task generation time"

    # *Something* must be variable across instances of the meta-task
    if np.any(stims>99) or np.any(np.array(rules)>99) :
        OK = True

# OK = True
print("PROBAUSESPECIALSTATE:", PROBAUSESPECIALSTATE, "PROBAUSEPROBABILISTICREWARDS:", 
                PROBAUSEPROBABILISTICREWARDS, "PROBAUSEVARREWARDPROB:", PROBAUSEVARREWARDPROB)
print("Special States' ranges:", specialstatesranges)
print("Transition table:\n", T)
print("Reward rules (Old state, new state, action taken, probability, value):\n", rules)
print("Stimuli:\n", stims)
