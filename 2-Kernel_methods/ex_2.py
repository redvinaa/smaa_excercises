import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
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


bounds = Bounds(np.zeros((n,)), np.full((n,), np.inf))
constraints = LinearConstraint(Y, 0, 0)
res = minimize(fun, np.zeros((n,)), bounds=bounds, constraints=constraints)

if not res.success:
	quit('ERROR: ' + res.message)

alpha = res.x

idx = np.argwhere(alpha!=0).flatten()[0]
x = X[idx]
y = Y[idx]

w = np.sum([a*y*x for a, y, x in zip(alpha, Y, X)], axis=0)
b = y - kernel(w, x)
b = 0

print(f'b = {b}')

#  alpha_outer = cp.vstack([alpha*a for a in alpha])
#  outer       = cp.multiply(alpha_outer, K_Y)
#  obj         = cp.sum(alpha) - 1/2*eq
#  obj         = cp.Maximize(obj)
## ^^ this is how it would be done but this isn't DCP

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

def f(x, y):
	x_vec = np.array([x, y])
	return np.sum([alpha[k]*Y[k]*kernel(x_vec, X[k]) for k in range(p__n)]) + b

## }}}


## d) Plots {{{

s = 20 # marker size
loc = 2 # legend location
l=1.2
extent = (-p__R*l, p__R*l, -p__R*l, p__R*l)

pts_a_x = a__points[a__class== 1][:,0]
pts_a_y = a__points[a__class== 1][:,1]
pts_b_x = a__points[a__class==-1][:,0]
pts_b_y = a__points[a__class==-1][:,1]
plt.scatter(pts_a_x, pts_a_y, label='Sample: class a', color='black')
plt.scatter(pts_b_x, pts_b_y, label='Sample: class b', \
	edgecolors='black', facecolors='none')

m = p__n # resolution of image
X_ = np.linspace(extent[0], extent[1], m)
Z_ = np.array([np.array([f(x1, x2) for x1 in X_]) for x2 in X_])

ax = plt.gca()
im = ax.matshow(Z_,cmap=cm.RdBu, extent=extent)
cset = plt.contour(Z_,[-10, -1, 0, 1, 10],linewidths=2,colors='black', extent=extent)
plt.clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
plt.colorbar(im)

plt.legend(loc=loc)
plt.grid()
plt.savefig('../figures/ex_II_2_plots_1.pdf')
plt.show()

## }}}
