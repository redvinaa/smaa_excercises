p__n = 200 # play this many episodes
p__n_per_set = 3 # average this many runs per set of parameters

p__ncols = 12
p__nrows = 4
p__disc = .9 # discount factor

p__eps = 0.2
p__tau = 0.1 # Boltzmann temperature

p__policy = 'e-greedy'

p__c1  = [None, None, None, None] # step-size coeff
p__c2  = [None, None, None, None] # momentum term coeff
p__pol = [None, None, None, None] # polyak averaging window

#  c1 = 2, c2 = 2, pol = 2

p__c1  = [None, 1, 2, 2, 3]
p__c2  = [None, 3, 2, 3, 1]
p__pol = [None, 2, 2, 2, 2]
