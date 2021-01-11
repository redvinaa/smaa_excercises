import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cvxpy as cp
from cliffwalking import CliffWalkingEnv
from time import sleep
from ex_1_param import *


## Misc {{{

env = CliffWalkingEnv()

def value_to_policy(V):
	policy = np.empty(shape=(env.observation_space.n,))
	for S in range(env.observation_space.n):
		policy[S] = np.argmax([V[S_next] for S_next in np.array(model[S, :, 1], dtype=int)])
	return policy

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


## a) Generating model {{{

# model: (S, A) -> (R, S_)
#
#      R  S_
#   S  x  x
#   A  x  x

# generate or load model
try:
	if not p__use_saved_model:
		raise FileNotFoundError()
	model = np.load('model.npy')
	print('Loaded model')
except FileNotFoundError:
	model = np.zeros(shape=(env.observation_space.n, env.action_space.n, 2))
	for i in range(p__model_n):
		S = env.reset()
		
		done = False
		while not done:
			A = env.action_space.sample()
			S_, R, done, _ = env.step(A)
			model[S, A] = np.array([R, S_])
			S = S_

	np.save('model.npy', model)
	print('Generated model')

## }}}


## b) Linear Programming {{{

lambd = cp.Variable(env.observation_space.n)

obj   = cp.Minimize(cp.sum(lambd))
constr = [lambd[S] >= model[S, A, 0] + p__disc * lambd[int(model[S, A, 1])] \
	for S in range(env.observation_space.n) for A in range(env.action_space.n)] + \
	[lambd[1] == 0] # final state has a (given) zero value, don't minimize for that

prob = cp.Problem(obj, constr)
prob.solve()

V_LP = lambd.value

np.save('LP_value_function.npy', V_LP)

plt.matshow(V_to_scene(V_LP), cmap='Greys')
plt.savefig('../figures/ex_III_1_plots_LP.pdf')
plt.show()
plt.close()

## }}}


## c) Value Iteration {{{

dist_VI   = np.empty((p__n_per_policy, p__iter,))

for sub_it in range(p__n_per_policy):

	V_VI      = np.zeros(shape=(env.observation_space.n,))

	for i in range(p__iter):
		for S in range(env.observation_space.n):
			if S == 1:
				continue
			V_VI[S] = np.max([model[S, A, 0] + p__disc * V_VI[int(model[S, A, 1])] \
	##                        ^^next immediate reward    ^^ value of next state
				for A in range(env.action_space.n)])

		dist_VI[sub_it, i] = np.linalg.norm(V_LP-V_VI)

dist_VI = np.average(dist_VI, axis=0)

plt.close()

## }}}


## d) Policy Iteration {{{

dist_PI = np.empty((p__n_per_policy, p__iter,))

for sub_it in range(p__n_per_policy):

	policy  = np.random.choice([0, 1, 2, 3], size=env.observation_space.n)
	V_PI    = np.empty((env.observation_space.n,))

	for i in range(p__iter):

		for S in range(env.observation_space.n):
			if S == 1:
				continue
			V_PI[S] = model[S, policy[S], 0] + p__disc * V_PI[int(model[S, policy[S], 1])]
	##                ^^ immediate reward                ^^ discounted value of next state

		for S in range(env.observation_space.n):
			if S == 1:
				continue
			policy[S] = np.argmax([model[S, A, 0] + \
			##                     ^^ next reward
				p__disc * V_PI[int(model[S, A, 1])] for A in range(env.action_space.n)])
			##                 ^^ next state

		dist_PI[sub_it, i] = np.linalg.norm(V_LP-V_PI)

dist_PI = np.average(dist_PI, axis=0)

plt.close()

## }}}


## e) Plots {{{

plt.plot(np.linspace(1, p__iter+1, p__iter), dist_VI, label='Value Iteration')
plt.plot(np.linspace(1, p__iter+1, p__iter), dist_PI, label='Policy Iteration')
plt.legend()
plt.grid()
plt.savefig('../figures/ex_III_1_plots_dist.pdf')
plt.show()
plt.close()

## }}}

## play an episode with policy
#  policy = value_to_policy(V_VI)
#  S = env.reset()
#  done = False
#  R_sum = 0
#  while not done:
#      A = policy[S]
#      S, R, done, _ = env.step(A)
#      R_sum += R
#      print(f'action: {env.action_names[A]}, reward: {R}, total rewards: {R_sum}')
#      env.render()
#      sleep(.5)
