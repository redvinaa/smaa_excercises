#! /usr/bin/env python3
# file: II_2/ex_II_2.py

import numpy as np
from scipy.linalg import qr, inv
import cvxpy as cp
import matplotlib.pyplot as plt
from ex_2_param import *


## a) Sample generation {{{

a__class = np.random.choice([-1, 1], size=p__n)

a__points = np.empty(shape=(p__n, 2))
for i, c in enumerate(a__class):
	if c == 1:
		a__points[i] = p__cov_a @ np.random.standard_normal(size=2)
	else:
		r     = p__R + np.random.standard_normal()*p__cov_b
		theta = np.random.uniform(low=0, high=2*np.pi)

		a__points[i] = np.array([r*np.cos(theta), r*np.sin(theta)])

## }}}


## b) Kernel implementation {{{

def kernel(x, y, mode='gaussian', params=None):
	if mode=='linear':
		return x @ y
	if mode=='polynomial':
		assert('c' in params)
		assert('p' in params)
		c = params['c']
		p = params['p']
		return (x @ y + c) ** p
	if mode=='laplacian':
		assert('sigma' in params)
		sigma = params['sigma']
		return np.exp(-np.linalg.norm(x-y, ord=1)/sigma**2)
	else:
		assert('sigma' in params)
		sigma = params['sigma']
		return np.exp(-np.linalg.norm(x-y, ord=2)/sigma**2)

## }}}


## c) Optimization {{{

X = a__points
Y = a__class

n = p__n
lambd = p__lambd

a_p = cp.Variable(n, nonneg=True)
a_n = cp.Variable(n, nonneg=True)

K = np.empty(shape=(n, n))
for i in range(n):
	for j in range(n):
		K[i, j] = kernel(X[i], X[j], mode=p__mode, params=p__params)

obj = Y @ (a_p-a_n) - 1/2 * cp.pos((a_p-a_n)) @ cp.pos((a_p-a_n)) - p__eps * cp.sum(a_p+a_n)

eq1 = cp.sum([(a_p[i]-a_n[i])**2 for i in range(n)])

eq = (a_p-a_n) @ K @ (a_p-a_n)
obj = Y @ (a_p-a_n) - 1/2 * eq1 - p__eps * cp.sum(a_p+a_n)

constr = [a <= lambd for a in np.append(a_p, a_n)]
constr.append(cp.sum(a_p-a_n) == 0)

prob = cp.Problem(cp.Maximize(obj), constr)
prob.solve()

a_p = a_p.value
a_n = a_n.value
w = (a_p-a_n) @ X

s_b = 0 # TODO
def f_optimal(x):
	return np.sum([(a_p_k-a_n_k) @ kernel(x, x_k) for a_p_k, a_n_k, x_k in zip(a_p, a_k, X)]) + s_b

## }}}


## d) Plots {{{

s = 20 # marker size
loc = 2 # legend location

pts_a_x = a__points[a__class== 1][:,0]
pts_a_y = a__points[a__class== 1][:,1]
pts_b_x = a__points[a__class==-1][:,0]
pts_b_y = a__points[a__class==-1][:,1]
plt.scatter(pts_a_x, pts_a_y, label='Sample: class a', color='black')
plt.scatter(pts_b_x, pts_b_y, label='Sample: class b', \
	edgecolors='black', facecolors='none')

plt.legend(loc=loc)
plt.grid()
plt.show()

## }}}
