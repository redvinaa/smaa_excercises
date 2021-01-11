import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from cliffwalking import CliffWalkingEnv
from ex_1_param import *


## a) Misc {{{

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

def policiy_soft_max(S, Q, tau=p__tau):
	assert(tau>0)
	den = np.sum([np.exp(Q[S, a]/tau) for a in range(env.action_space.n)])
	p = [np.exp(Q[S, a]/tau)/den for a in range(env.action_space.n)]
	return np.random.choice([0, 1, 2, 3], p=p)

policies = {'random': policy_random, 
			'e-greedy': policy_e_greedy,
			'soft-max': policiy_soft_max}

policy = policies[p__policy]

## }}}


## b) Testing policies {{{

pname   = p__policy
fig, ax = plt.subplots(ncols=p__nplots, nrows=len(p__c1))
X       = np.linspace(1, p__n, p__n)
dists   = np.zeros(shape=(len(p__c1), p__n))
rewards = np.zeros(shape=(len(p__c1), p__n))


# same process for all policies
for idx_c, (c1, c2, pol) in enumerate(zip(p__c1, p__c2, p__pol)):
	Q      = np.zeros(shape=(env.observation_space.n, env.action_space.n))
	Q_prev = np.zeros(shape=(env.observation_space.n, env.action_space.n))
	visits = np.zeros(shape=(env.observation_space.n, env.action_space.n), dtype=np.int)
	if pol:
		Q_past = np.zeros(shape=(env.observation_space.n, env.action_space.n, pol))

	np.random.seed(0)

	# iterating through p__n episodes
	for it in range(p__n):
		
		# playing one episode
		S = env.reset()
		done = False
		while not done:
			A = policy(S, Q)
			S_, R, done, _ = env.step(A)
			rewards[idx_c, it] += R

			visits[S, A] += 1
			vis = visits[S, A]

			if c1:
				gamma = c1 / vis
			else:
				gamma = 1 / (1 + it)

			if vis > 1 and c2:
				beta = c2 / vis
			else:
				beta = 0

			#  if pol:
			#      gammas[S, A, (vis-1)%pol] = gamma
			#      if vis > 4:
			#          gamma = np.average(gammas[S, A, :])

			# updating Q
			V_next = np.max([Q[S_, a] for a in range(env.action_space.n)])
			Q_curr = Q[S, A]
			Q[S, A] = (1 - gamma) * Q_curr + gamma * (R + p__disc * V_next) + beta * (Q_curr - Q_prev[S, A])
			Q_prev[S, A] = Q_curr


			S = S_

		dists[idx_c, it] = dist_V(Q)

		if it in p__plot_at:
			print(f'showing fig: c1 = {c1}, c2 = {c2}, pol = {pol}, it = {it}')

			idx_it = np.argwhere(np.array(p__plot_at) == it).flatten()[0]
			ax[idx_c, idx_it].matshow(V_to_scene(to_V(Q)), cmap='Greys', norm=Normalize())
			ax[idx_c, idx_it].set_title(f'c1 = {c1}, c2 = {c2}, pol = {pol}, it = {it}')
			#  plt.savefig(f'../figures/ex_IV_1_plots_{pname}_{it}.pdf')
			#  plt.show()

plt.tight_layout()
plt.show()

## }}}


## c) Plots {{{

fig, ax = plt.subplots(nrows=2)

for i, (c1, c2, pol) in enumerate(zip(p__c1, p__c2, p__pol)):
	ax[0].plot(X, dists[i], label=f'c1={c1}, c2={c2}, pol = {pol}')

	ax[1].plot(X, rewards[i], label=f'c1={c1}, c2={c2}, pol = {pol}')

ax[0].grid()
ax[0].legend()
ax[0].set_title('Value function distance')
ax[1].set_title('Total rewards')
ax[1].grid()
ax[1].legend()

plt.savefig(f'../figures/ex_IV_1_plots_dists_rewards.pdf')
plt.show()

## }}}
