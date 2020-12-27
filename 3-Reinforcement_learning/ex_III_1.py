# file: III_1/ex_III_1.py

import numpy as np
import matplotlib.pyplot as plt
from cliffwalking_my import CliffWalkingEnv
from time import sleep
from ex_III_1_parameters import *


## a) Generating model {{{

env = CliffWalkingEnv()

# model should be a function like
# f: (S, A) -> (R, S_)

#      R  S_
#   S  x  x
#   A  x  x

model = np.zeros(shape=(env.observation_space.n, env.action_space.n, 2))

for i in range(p__model_n):
	S = env.reset()
	steps = 0
	
	done = False
	while not done:
		A = env.action_space.sample()
		S_, R, done, _ = env.step(A)
		model[S, A] = np.array([R, S_])
		# env is not stochastic, don't need step size

		S = S_
		steps += 1

print('Generated model')

## }}}


## b) Value Iteration {{{

V  = np.zeros(shape=(env.observation_space.n,))

policy_greedy = np.empty(shape=(env.observation_space.n))
policy        = np.empty(shape=(env.observation_space.n))

fig, ax = plt.subplots(ncols=p__nplots, nrows=3)

for i in range(p__iter):
	for S in range(env.observation_space.n):
		V[S] = np.max([model[S, a, 0] + p__disc * V[int(model[S, a, 1])] for a in range(env.action_space.n)])
##                    ^^next immediate reward     ^^ value of next state

	if (i+1)%p__plot_every == 0:
		ax[i//p__plot_every, 0].matshow(np.flip(V.reshape(p__nrows, p__ncols), axis=0))
		ax[i//p__plot_every, 0].set_title(f'Value function, iter: {i+1}')


		for S in range(env.observation_space.n):
			A_next = np.argmax([V[S_next] for S_next in np.array(model[S, :, 1], dtype=int)])
			S_next = model[S, A_next, 1]
			R_next = model[S, A_next, 0]

			policy_greedy[S] = np.argmax([model[S, a, 0] + \
				p__disc * V[int(model[S, a, 1])] for a in range(env.action_space.n)])
			policy[S]        = A_next

			ax[i//p__plot_every, 1].matshow(np.flip(policy.reshape(p__nrows, p__ncols), axis=0))
			ax[i//p__plot_every, 1].set_title(f'Non-greedy policy after {i+1} iterations')

			ax[i//p__plot_every, 2].matshow(np.flip(policy_greedy.reshape(p__nrows, p__ncols), axis=0))
			ax[i//p__plot_every, 2].set_title(f'Greedy policy after {i+1} iterations')

plt.show()


S = env.reset()
done = False
R_sum = 0
while not done:
	A = policy[S]
	S, R, done, _ = env.step(A)
	R_sum += R
	print(f'action: {env.action_names[A]}, reward: {R}, total rewards: {R_sum}')
	env.render()
	sleep(.5)

## }}}


## c) Policy Iteration {{{

## }}}


## d) Linear Programming (optimal value function) {{{

## }}}
