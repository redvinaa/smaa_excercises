import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cvxpy as cp
from cliffwalking_my import CliffWalkingEnv
from time import sleep
from ex_2_param import *


## Misc {{{

def value_to_policy(V):
	policy = np.empty(shape=(env.observation_space.n,))
	for S in range(env.observation_space.n):
		policy[S] = np.argmax([V[S_next] for S_next in np.array(model[S, :, 1], dtype=int)])

	return policy

in2cm = 1/2.54

## }}}


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


## b) Linear Programming {{{

lambd = cp.Variable(env.observation_space.n)

obj   = cp.Minimize(cp.sum(lambd))
constr = [lambd[S] >= model[S, A, 0] + p__disc * lambd[int(model[S, A, 1])] for S in range(env.observation_space.n) for A in range(env.action_space.n)]

prob = cp.Problem(obj, constr)
prob.solve()

V_LP = lambd.value

scene = np.flip(V_LP.reshape(p__nrows, p__ncols), axis=0)
plt.imshow(scene, cmap='Greys', norm=Normalize())
#  plt.savefig('../figures/ex_III_1_plots_1.pdf')
#  plt.show()
plt.close()

## }}}


## c) Value Iteration {{{

V_VI      = np.zeros(shape=(env.observation_space.n))
b__policy = np.empty(shape=(env.observation_space.n))
dist_VI   = np.empty(shape=(p__iter))

fig, ax = plt.subplots(ncols=2, nrows=p__nplots)

idx = 0
for i in range(p__iter):
	for S in range(env.observation_space.n):
		V_VI[S] = np.max([model[S, A, 0] + p__disc * V_VI[int(model[S, A, 1])] for A in range(env.action_space.n)])
##                       ^^next immediate reward     ^^ value of next state

	dist_VI[i] = np.linalg.norm(V_LP-V_VI)

	if i in p__plot_at:
		b__policy = value_to_policy(V_VI)

		scene = np.flip(V_VI.reshape(p__nrows, p__ncols), axis=0)
		ax[idx, 0].matshow(scene, cmap='Greys', norm=Normalize())
		ax[idx, 0].set_title(f'Value function, iter: {i}')
                 
		scene = np.flip(b__policy.reshape(p__nrows, p__ncols), axis=0)
		ax[idx, 1].imshow(scene, cmap='Greys', vmin=0, vmax=3)

		ax[idx, 1].set_title(f'Policy, iter: {i}')
		# after enough iterations, the non-greedy policy is equal to the greedy policy

		idx += 1

plt.tight_layout()
#  plt.savefig('../figures/ex_III_1_plots_1.pdf')
#  plt.show()
plt.close()

## }}}


## d) Policy Iteration {{{

policy = np.random.choice([0, 1, 2, 3], size=env.observation_space.n)
V_PI   = np.empty(shape=(env.observation_space.n))
dist_PI   = np.empty(shape=(p__iter))

fig, ax = plt.subplots(ncols=2, nrows=p__nplots)

idx = 0
for i in range(p__iter):

	for S in range(env.observation_space.n):
		V_PI[S] = model[S, policy[S], 0] + p__disc * V_PI[int(model[S, policy[S], 1])]
##                ^^ immediate reward                ^^ discounted value of next state

	for S in range(env.observation_space.n):
		policy[S] = np.argmax([model[S, A, 0] + \
			p__disc * V_PI[int(model[S, A, 1])] for A in range(env.action_space.n)])

	dist_PI[i] = np.linalg.norm(V_LP-V_PI)

	if i in p__plot_at:
		b__policy = value_to_policy(V_PI)

		scene = np.flip(V_PI.reshape(p__nrows, p__ncols), axis=0)
		ax[idx, 0].imshow(scene, cmap='Greys', norm=Normalize())
		ax[idx, 0].set_title(f'Value function, iter: {i}')
                 
		scene = np.flip(policy.reshape(p__nrows, p__ncols), axis=0)
		ax[idx, 1].imshow(scene, cmap='Greys', vmin=0, vmax=3)
		ax[idx, 1].set_title(f'Policy, iter: {i}')
		# after enough iterations, the non-greedy policy is equal to the greedy policy

		idx += 1

plt.tight_layout()
plt.savefig('../figures/ex_III_1_plots_2.pdf')
#  plt.show()
plt.close()

## }}}


## e) Plots {{{

plt.plot(np.linspace(1, p__iter+1, p__iter), dist_VI, label='Value Iteration')
plt.plot(np.linspace(1, p__iter+1, p__iter), dist_PI, label='Policy Iteration')
plt.legend()
plt.grid()
#  plt.savefig('../figures/ex_III_1_plots_3.pdf')
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
