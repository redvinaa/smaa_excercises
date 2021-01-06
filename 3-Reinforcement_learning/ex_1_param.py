p__model_n = 10 # play this many episodes for model gen

p__ncols = 12
p__nrows = 4
p__disc = .9 # discount factor

p__plot_at = [1, 2, 5, 12]
p__iter = p__plot_at[-1]+1 # play this many iters for learning
p__nplots = len(p__plot_at)

p__use_saved_model = True
