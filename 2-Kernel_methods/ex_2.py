import numpy as np
#  from scipy.optimize import minimize, Bounds, LinearConstraint
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
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
		return np.exp(-(np.linalg.norm(x-y)/sigma)**2)

## }}}


## c) Optimization {{{

X = a__points
Y = a__class
n = p__n

K = np.empty(shape=(n, n))
for i in range(n):
	for j in range(n):
		K[i, j] = kernel(X[i], X[j])

alpha = cp.Variable(n)

obj = cp.sum(alpha) - 1/2*cp.quad_form(cp.multiply(alpha, Y), K)
obj = cp.Maximize(obj)

constr = [alpha[k] >= 0 for k in range(n)]
constr += [cp.sum(cp.multiply(alpha, Y)) == 0]

prob = cp.Problem(obj, constr)
prob.solve()

alpha = alpha.value
alpha = np.around(alpha, 10)
print(f'alpha: \n{alpha}')
print(f'{np.min(np.abs(alpha))}')

idx = np.argwhere(np.abs(alpha) != 0).flatten()[0]
w = np.sum([a*y*x for a, y, x in zip(alpha, Y, X)], axis=0)
b = Y[idx] - np.sum([alpha[k]*Y[k]*kernel(X[idx], X[k]) for k in range(p__n)])
print(f'w = {w}')
print(f'b = {b}')

def f(x, y):
	x_vec = np.array([x, y])
	return np.sum([alpha[k]*Y[k]*kernel(x_vec, X[k]) for k in range(p__n)]) + b

## }}}


## d) Plots {{{

s = 20 # marker size
loc = 2 # legend location
l=1.2
extent = (-p__R*l, p__R*l, -p__R*l, p__R*l)

## black & white image
pts_a_x = a__points[a__class== 1][:,0]
pts_a_y = a__points[a__class== 1][:,1]
pts_b_x = a__points[a__class==-1][:,0]
pts_b_y = a__points[a__class==-1][:,1]
plt.scatter(pts_a_x, pts_a_y, label='sample A', color='red')
plt.scatter(pts_b_x, pts_b_y, label='sample B', \
	edgecolors='blue', facecolors='none')

m = p__n # resolution of image
X_ = np.linspace(extent[0], extent[1], m)
Z_ = np.array([np.array([np.sign(f(x1, x2)) for x1 in X_]) for x2 in X_])

ax = plt.gca()
im = ax.imshow(Z_, cmap='Greys', extent=extent)

plt.legend(loc=loc)
plt.grid()
plt.savefig('../figures/ex_II_2_plots_1.pdf')
plt.show()

# color image
pts_a_x = a__points[a__class== 1][:,0]
pts_a_y = a__points[a__class== 1][:,1]
pts_b_x = a__points[a__class==-1][:,0]
pts_b_y = a__points[a__class==-1][:,1]
plt.scatter(pts_a_x, pts_a_y, label='sample A', color='black')
plt.scatter(pts_b_x, pts_b_y, label='sample B', \
	edgecolors='black', facecolors='none')

m = p__n # resolution of image
X_ = np.linspace(extent[0], extent[1], m)
Z_ = np.array([np.array([f(x1, x2) for x1 in X_]) for x2 in X_])

ax = plt.gca()
im = ax.imshow(Z_, cmap=cm.RdBu, extent=extent)
plt.colorbar(im)

plt.legend(loc=loc)
plt.grid()
plt.savefig('../figures/ex_II_2_plots_2.pdf')
plt.show()

## }}}
