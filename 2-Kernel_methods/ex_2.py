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
		a__points[i] = p__cov_a @ np.random.standard_normal(size=2) + p__mean
	else:
		r     = p__R + np.random.standard_normal()*p__cov_b
		theta = np.random.uniform(low=0, high=2*np.pi)

		a__points[i] = np.array([r*np.cos(theta), r*np.sin(theta)]) + p__mean

## }}}


## b) Kernel implementation {{{

def kernel(x, y, mode=p__mode, params=p__params):
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
n     = p__n

K = np.empty(shape=(n, n))
K_Y = np.empty(shape=(n, n))
for i in range(n):
	for j in range(n):
		K[i, j] = kernel(X[i], X[j])
		K_Y[i, j] = K[i, j] * Y[i] * Y[j]

def fun(alpha):
	eq = np.sum(alpha) - 1/2 * np.sum([alpha[k]*alpha[m]*Y[k]*Y[m]*K[k,m] \
		for k in range(n) for m in range(n)])
	return -eq

from scipy.optimize import minimize, Bounds, LinearConstraint

bounds = Bounds(np.zeros((n,)), np.full((n,), np.inf))
constraints = LinearConstraint(Y, 0, 0)
res = minimize(fun, np.zeros((n,)), bounds=bounds, constraints=constraints)

if not res.success:
	quit('ERROR: ' + res.message)

alpha = res.x
print(f'alpha: {alpha}')

idx = np.argwhere(alpha!=0).flatten()[0]
print(f'idx = {idx}')

w = np.sum([a*y*x for a, y, x in zip(alpha, Y, X)], axis=0)
print(f'w = {w}')

x = X[idx]
y = Y[idx]
b = y - kernel(w, x)

print(f'b = {b}')

#  alpha_outer = cp.vstack([alpha*a for a in alpha])
#  outer       = cp.multiply(alpha_outer, K_Y)
#  obj         = cp.sum(alpha) - 1/2*eq
#  obj         = cp.Maximize(obj)
## ^^ this is how it would be done if cvxpy wasn't a piece of shit

#  constr = [lambd >= alpha[k] for k in range(n)]
#  constr = [alpha[k] >= 0 for k in range(n)]
#  constr.append(cp.sum([alpha[k]*Y[k] for k in range(n)]) == 0)

#  prob = cp.Problem(obj, constr)
#  prob.solve()

#  try:
#      prob.solve()
#  except cp.error.DCPError:
#      print('DCPERROR')
#  quit()


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

n = 100
X = np.random.uniform(-10, 20, size=2*n).reshape((-1, 2))
Y = np.array([kernel(w, x) + b for x in X])

pts_a_x = X[Y>=0][:, 0]
pts_a_y = X[Y>=0][:, 1]
pts_b_x = X[Y<0][:, 0]
pts_b_y = X[Y<0][:, 1]
plt.scatter(pts_a_x, pts_a_y, color='red', alpha=.3, label='Class a')
plt.scatter(pts_b_x, pts_b_y, color='blue', alpha=.3, label='Class b')

plt.legend(loc=loc)
plt.grid()
plt.savefig('../figures/ex_II_2_plots_1.pdf')
plt.show()

## }}}
