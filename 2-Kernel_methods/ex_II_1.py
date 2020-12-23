#! /usr/bin/env python3
# file: II_1/ex_II_1.py

import numpy as np
from scipy.linalg import qr, inv
import cvxpy as cp
import matplotlib.pyplot as plt
from ex_II_1_parameters import *


## a) Sample generation {{{

# classification (y values) of training data
a__class = np.random.choice([-1, 1], size=p__n)

# x values of training data
a__points = []

for i in range(3): # i: dataset idx
	a__points.append(np.empty(shape=(p__n, 2)))

	p__m_a[i]   = np.array(p__m_a[i])
	p__m_b[i]   = np.array(p__m_b[i])
	p__cov_a[i] = np.array(p__cov_a[i])
	p__cov_b[i] = np.array(p__cov_b[i])

	for j, c in enumerate(a__class): # j: a__class[j]
		if c == 1:
			a__points[i][j] = np.random.multivariate_normal(p__m_a[i], p__cov_a[i])
			a__points[i][j] = np.random.multivariate_normal(p__m_a[i], p__cov_a[i])
			a__points[i][j] = np.random.multivariate_normal(p__m_a[i], p__cov_a[i])
		else:
			a__points[i][j] = np.random.multivariate_normal(p__m_b[i], p__cov_b[i])
			a__points[i][j] = np.random.multivariate_normal(p__m_b[i], p__cov_b[i])
			a__points[i][j] = np.random.multivariate_normal(p__m_b[i], p__cov_b[i])

## }}}


## b) Implementing classifiers {{{

Y = a__class
m = len(Y)
n = 2

## Soft margin SVC {{{
# Part-1-Statistical-Learning.pdf: 84 p.

W_SMSVC = np.empty(shape=(3, 2))
b_SMSVC = np.empty(shape=(3, 1))
for i in range(3):

	X = a__points[i]

	w     = cp.Variable(n)
	b     = cp.Variable()
	lambd = cp.Parameter(pos=True)
	Eps   = cp.Variable(m, nonneg=True)

	obj   = 1/2 * cp.norm(w, 2) + lambd * cp.sum(Eps)
	constr = [y * (w @ x + b) >= 1-eps for y, x, eps in zip(Y, X, Eps)]
	prob = cp.Problem(cp.Minimize(obj), constr)

	lambd.value = p__lambd
	res = prob.solve()

	W_SMSVC[i] = w.value
	b_SMSVC[i] = b.value

print(f'W_SMSVC = {W_SMSVC}')
print(f'b_SMSVC = {b_SMSVC}')

## }}}

## Least Squares SVM {{{

W_LSSVM = np.empty(shape=(3, 2))
b_LSSVM = np.empty(shape=(3, 1))
for i in range(3):

	X = a__points[i]

	w     = cp.Variable(n)
	b     = cp.Variable()
	lambd = cp.Parameter(pos=True)
	Eps   = cp.Variable(m)

	obj   = 1/2 * cp.norm(w, 2) + lambd * cp.sum([e**2 for e in Eps])
	constr = [y * (x @ w + b) == 1-eps for y, x, eps in zip(Y, X, Eps)]
	prob = cp.Problem(cp.Minimize(obj), constr)

	lambd.value = p__lambd
	res = prob.solve()

	W_LSSVM[i] = w.value
	b_LSSVM[i] = b.value

print(f'W_LSSVM = {W_LSSVM}')
print(f'b_LSSVM = {b_LSSVM}')

## }}}

## Nearest centroid {{{

ncc_center_a = np.empty(shape=(3, 2))
ncc_center_b = np.empty(shape=(3, 2))
for i in range(3):

	X = a__points[i]

	ncc_center_a[i] = np.average(X[a__class== 1], axis=0)
	ncc_center_b[i] = np.average(X[a__class==-1], axis=0)

print(f'ncc_center_a = {ncc_center_a}')
print(f'ncc_center_b = {ncc_center_b}')

## }}}

## }}}


## c) Plots and evaluation {{{

s = 20. # marker size
loc = 2 # legend location

for i in range(3):

	fig, ax = plt.subplots(ncols=3)
	for j in range(3):
		ax[j].set_aspect(1, 'box')

	# Sample {{{
	points_a_x1 = a__points[i][a__class==1][:,0]
	points_a_x2 = a__points[i][a__class==1][:,1]
	points_b_x1 = a__points[i][a__class==-1][:,0]
	points_b_x2 = a__points[i][a__class==-1][:,1]

	for j in range(3):
		ax[j].scatter(points_a_x1, points_a_x2, label='Sample: class a', \
			edgecolors='black', facecolors='none', s=s)
		ax[j].scatter(points_b_x1, points_b_x2, label='Sample: class b', facecolors='black', s=s)

	xl, xr = ax[i].get_xlim()
	yl, yr = ax[i].get_ylim()
	# }}}

	# SMSVC {{{
	w = W_SMSVC[i]
	b = b_SMSVC[i]

	c = 'purple'

	X = np.linspace(-10, 10, 2)
	Y = - w[0]*X/w[1] - b/w[1]
	ax[0].plot(X, Y, color=c)

	Y = - w[0]/w[1]*X - b/w[1] + 1/w[1]
	ax[0].plot(X, Y, alpha=.2, color=c)

	Y = - w[0]/w[1]*X - b/w[1] - 1/w[1]
	ax[0].plot(X, Y, alpha=.2, color=c)

	ax[0].set_title('Soft Margin SVC')
	# }}}

	# LSSVM {{{
	w = W_LSSVM[i]
	b = b_LSSVM[i]

	c = 'orange'

	X = np.linspace(-10, 10, 2)
	Y = - w[0]*X/w[1] - b/w[1]
	ax[1].plot(X, Y, color=c)

	Y = - w[0]/w[1]*X - b/w[1] + 1/w[1]
	ax[1].plot(X, Y, alpha=.2, color=c)

	Y = - w[0]/w[1]*X - b/w[1] - 1/w[1]
	ax[1].plot(X, Y, alpha=.2, color=c)

	ax[1].set_title('Least Squares SVM')
	# }}}

	# NCC {{{
	s_ = 8*s

	c_a = ncc_center_a[i]
	c_b = ncc_center_b[i]

	ax[2].scatter(c_a[0], c_a[1], label='Sample mean (a)', \
		edgecolors='green', facecolors='none', s=s_)
	ax[2].scatter(c_b[0], c_b[1], label='Sample mean (b)', \
		facecolors='green', s=s_)

	p = (c_a+c_b) / 2
	v_norm = c_a - c_b

	# v_norm * (x - p) = 0  ->  v_norm[0]*(x0-p0) + v_norm[1]*(x1-p1) = 0
	X = np.linspace(-10, 10, 2)
	Y = -v_norm[0]/v_norm[1]*(X-p[0]) + p[1]

	ax[2].plot(X, Y, color='green')

	ax[2].set_title('Nearest Centroid')
	# }}}

	for a in ax:
		a.legend(loc=loc, fancybox=True, framealpha=1)
		a.grid()
		a.set_xlabel('$x_1$')
		a.set_ylabel('$x_2$')
		a.set_xlim((xl, xr))
		a.set_ylim((yl, yr))

	fig = plt.gcf()
	fig.set_size_inches((20, 10), forward=False)
	#  fig.savefig(f'../figures/ex_II_1_plots_{i}.pdf')

	#  plt.show()
	plt.close()
# }}}

## }}}
