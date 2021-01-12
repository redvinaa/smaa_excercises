import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from cliffwalking import CliffWalkingEnv
from ex_2_param import *


## a) Misc {{{

env = CliffWalkingEnv()
V_LP = np.load('LP_value_function.npy') # optimal value-function

def to_V(Q): # get V from Q
	return np.max(Q, axis=1)

def dist_V(Q): # how far a given V is from the optimal (V_LP)
	return np.linalg.norm(V_LP - to_V(Q))

def V_to_scene(V): # place V values on the grid ("scene")
	# because you can't stay on the cliff, those blocks
	# are not states, so |V| < 4*12
	scene      = np.zeros((p__ncols*p__nrows,))
	scene[0]   = V[0]  # start
	scene[11]  = V[1]  # goal
	scene[12:] = V[2:] # other states
	scene      = np.flip(scene.reshape((p__nrows, p__ncols)), axis=0)
	return scene

## }}}


## a) Creating policies {{{

def policy_random(*args):
	return env.action_space.sample()

def policy_e_greedy(S, Q):
	ran = p__eps > np.random.uniform()
	if ran:
		return env.action_space.sample()
	else:
		return np.argmax(Q[S, :])

def policiy_soft_max(S, Q):
	assert(p__tau>0)
	den = np.sum([np.exp(Q[S, a]/p__tau) for a in range(env.action_space.n)])
	p = [np.exp(Q[S, a]/p__tau)/den for a in range(env.action_space.n)]
	return np.random.choice([0, 1, 2, 3], p=p)

policies = [policy_random, policy_e_greedy, policiy_soft_max]
pnames   = ['random',      'e-greedy',      'soft-max']

## }}}


## b) Testing policies {{{

X       = np.linspace(1, p__n, p__n)
dists   = {}
rewards = {}

# same process for all policies
for policy, pname in zip(policies, pnames):
	print(f'policy: {pname}')

	dists[pname]   = np.zeros((p__n_per_policy, p__n,))
	rewards[pname] = np.zeros((p__n_per_policy, p__n,))

	# averaging <p__n_per_policy> runs
	for sub_it in range(p__n_per_policy):

		Q = np.zeros((env.observation_space.n, env.action_space.n,))

		# iterating through p__n episodes
		for it in range(p__n):
			print(f'    it: {it}')
			lr = 1 / (1 + 0.05*it) # learning rate
			
			# playing one episode
			S = env.reset()
			done = False
			while not done:
				A = policy(S, Q)
				S_, R, done, _ = env.step(A)

				rewards[pname][sub_it, it] += R

				# updating Q
				V_next = np.max([Q[S_, a] for a in range(env.action_space.n)])
				Q[S, A] = (1 - lr) * Q[S, A] + lr * (R + p__disc * V_next)

				S = S_

			dists[pname][sub_it, it] = dist_V(Q)

	rewards[pname] = np.average(rewards[pname], axis=0)
	dists[pname]   = np.average(dists[pname],   axis=0)

## }}}


## c) Plots {{{

for pname in dists:
	plt.plot(X, dists[pname], label=pname)

plt.legend()
plt.grid()
plt.savefig(f'../figures/ex_III_2_plots_dists.pdf')
plt.show()

for pname in dists:
	if pname == 'random':
		plt.plot(X, rewards[pname]/100, label=pname+'/100', alpha=.4)
	else:
		plt.plot(X, rewards[pname], label=pname)

plt.legend()
plt.grid()
plt.savefig(f'../figures/ex_III_2_plots_rewards.pdf')
plt.show()

## }}}
