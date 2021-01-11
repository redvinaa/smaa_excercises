import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from cliffwalking import CliffWalkingEnv
from ex_1_param import *


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

policies = {'random': policy_random, 
			'e-greedy': policy_e_greedy,
			'soft-max': policiy_soft_max}

policy = policies[p__policy]

## }}}


def polyak_fun(n):
	return int(np.ceil(np.log(n+1)))
	return n
	return int(np.ceil(n/2))


## b) Testing policies {{{

pname   = p__policy
X       = np.linspace(1, p__n, p__n)
dists   = np.zeros((p__n_per_set, len(p__c1), p__n))
rewards = np.zeros((p__n_per_set, len(p__c1), p__n))

for sub_it in range(p__n_per_set):
	print(f'per_set: {sub_it}')

	# same process for all policies
	for idx_c, (c1, c2, pol) in enumerate(zip(p__c1, p__c2, p__pol)):
		print(f'parameter set: {idx_c}')
		
		Q      = np.zeros((env.observation_space.n, env.action_space.n))
		Q_prev = np.zeros((env.observation_space.n, env.action_space.n))
		visits = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.int)

		if pol:
			Q_pol  = np.zeros((p__n, env.observation_space.n, env.action_space.n)) # save Q values

		# iterating through p__n episodes
		for it in range(p__n):
			# playing one episode
			S = env.reset()
			done = False
			while not done:
				A = policy(S, Q)
				S_, R, done, _ = env.step(A)
				rewards[sub_it, idx_c, it] += R

				visits[S, A] += 1
				vis = visits[S, A]

				if c1:
					gamma = c1 / vis
				else:
					gamma = 1 / (it + 1)

				if vis > 1 and c2:
					beta = c2 / vis
				else:
					beta = 0

				# updating Q
				V_next = np.max([Q[S_, a] for a in range(env.action_space.n)])
				Q_curr = Q[S, A]
				Q[S, A] = (1 - gamma) * Q_curr + gamma * (R + p__disc * V_next) + beta * (Q_curr - Q_prev[S, A])
				Q_prev[S, A] = Q_curr

				S = S_
			if pol:
				Q_pol[it] = Q

				cw = polyak_fun(it) # current window

				if it == 0:
					dists[sub_it, idx_c, it] = dist_V(Q)
				elif it < cw:
					dists[sub_it, idx_c, it] = dist_V(np.average(Q_pol[:it+1], axis=0))
				else:
					dists[sub_it, idx_c, it] = dist_V(np.average(Q_pol[it+1-cw:it+1], axis=0))
			else:
				dists[sub_it, idx_c, it] = dist_V(Q)

dists   = np.average(dists,   axis=0)
rewards = np.average(rewards, axis=0)

## }}}


## c) Plots {{{

for i, (c1, c2, pol) in enumerate(zip(p__c1, p__c2, p__pol)):
	if i==0:
		plt.plot(X, dists[i],   label=f'c1={c1}, c2={c2}, pol={pol}', color='black')
	elif not pol:
		plt.plot(X, dists[i],   label=f'c1={c1}, c2={c2}, pol={pol}', ls='dashed')
	else:
		plt.plot(X, dists[i],   label=f'c1={c1}, c2={c2}, pol={pol}')
plt.grid()
plt.legend()
plt.savefig(f'../figures/ex_IV_1_plots_dists.pdf')
plt.show()

for i, (c1, c2, pol) in enumerate(zip(p__c1, p__c2, p__pol)):
	if i==0:
		plt.plot(X, rewards[i],   label=f'c1={c1}, c2={c2}, pol={pol}', color='black')
	elif not pol:
		plt.plot(X, rewards[i],   label=f'c1={c1}, c2={c2}, pol={pol}', ls='dashed')
	else:
		plt.plot(X, rewards[i],   label=f'c1={c1}, c2={c2}, pol={pol}')
plt.grid()
plt.legend()
plt.xlim([0, 50])
plt.savefig(f'../figures/ex_IV_1_plots_rewards.pdf')
plt.show()

## }}}
