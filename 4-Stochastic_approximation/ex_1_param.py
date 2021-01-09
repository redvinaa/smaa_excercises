p__plot_at = [1, 2, 5, 10, 30]
p__plot_at = [2, 6, 20]
p__n = p__plot_at[-1]+1 # play this many episodes
p__nplots = len(p__plot_at)

p__ncols = 12
p__nrows = 4
p__disc = .9 # discount factor

p__eps = 0.1
p__tau = 0.1 # Boltzmann temperature

p__policy = 'e-greedy'


p__c1  = [None, None, None, None]
p__c2  = [None, None, None, None]
p__pol = [None, None, None, None] # polyak averaging window

p__c1  = [None, 1,    None, 1]
p__c2  = [None, None, 1,    1]
#  p__pol = [1,    2,    3,    4]
