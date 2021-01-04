import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cvxpy as cp
from cliffwalking import CliffWalkingEnv
from time import sleep
from ex_2_param import *


## a) Setting up variables / functions {{{

env = CliffWalkingEnv()
V_LP = np.load('LP_value_function.npy') # optimal value-function
V_LP[1:11] = np.zeros(10) # technically, these are not states

#  scene = np.flip(V_LP.reshape((p__nrows, p__ncols)), axis=0)
#  plt.matshow(scene, cmap='Greys', norm=Normalize())
#  plt.show()

def to_V(Q): # get V from Q
	return np.max(Q, axis=1)

def dist_V(Q): # how far a given V is from the optimal (V_LP)
	return np.linalg.norm(V_LP - to_V(Q))

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

def policiy_soft_max(S, Q, tau=p__tau):
	assert(tau>0)
	den = np.sum([np.exp(Q[S, a]/tau) for a in range(env.action_space.n)])
	p = [np.exp(Q[S, a]/tau)/den for a in range(env.action_space.n)]
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

	dists[pname]   = np.empty((p__n,))
	rewards[pname] = np.zeros((p__n,))
	Q = np.zeros(shape=(env.observation_space.n, env.action_space.n))

	# iterating through p__n episodes
	for it in range(p__n):

		lr = 1 / (1 + it) # learning rate
		
		# playing one episode
		S = env.reset()
		done = False
		while not done:
			A = policy(S, Q)
			S_, R, done, _ = env.step(A)

			rewards[pname][it] += R

			# updating Q
			V_next = np.max([Q[S_, a] for a in range(env.action_space.n)])
			Q[S, A] = (1 - lr) * Q[S, A] + lr * (R + p__disc * V_next)

			S = S_

		dists[pname][it] = dist_V(Q)

		if it in p__plot_at:
			print(f'showing fig: pname = {pname}, it = {it}')
			scene = np.flip(to_V(Q).reshape((p__nrows, p__ncols)), axis=0)
			plt.matshow(scene, cmap='Greys', norm=Normalize())
			plt.savefig(f'../figures/ex_III_2_plots_{pname}_{it}.pdf')
			plt.show()

## }}}


## c) Plots {{{

for pname in dists:
	plt.plot(X, dists[pname], label=pname)

plt.legend()
plt.grid()
plt.savefig(f'../figures/ex_III_2_plots_{pname}_dists.pdf')
plt.show()

for pname in dists:
	if pname == 'random':
		plt.plot(X, rewards[pname]/100, label=pname+'/100', alpha=.4)
	else:
		plt.plot(X, rewards[pname], label=pname)

plt.legend()
plt.grid()
plt.savefig(f'../figures/ex_III_2_plots_{pname}_rewards.pdf')
plt.show()

## }}}
