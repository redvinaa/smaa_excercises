#! /usr/bin/env python3
# file: I_1/ex_I_1.py

import numpy as np
from scipy.linalg import qr, inv
import matplotlib.pyplot as plt
from basis_functions import poly_basis, sin_basis, exp_basis, Phi_gen
from SVD import pseudoinverse
from ex_I_1_parameters import *


## a) Sample generation {{{

def f(x):
	return pa__c * x * np.sin(pa__c * x)

a__X = np.random.uniform(0., 1., pa__n)
a__Y = np.array([f(x)+np.random.standard_normal() for x in a__X])

## }}}


## b) Linear regression estimate {{{

b__basis_fns    = {'poly': poly_basis, 'exp': exp_basis, 'sin': sin_basis}
b__chosen_basis = b__basis_fns[pb__basis]

b__Phi = Phi_gen(a__X, b__chosen_basis, pb__d, pb__sigma)

# c_ means cap/optimal
b__c_Theta = inv(b__Phi.T .dot(b__Phi)) .dot(b__Phi.T) .dot(a__Y)
b__Y_FIT = b__Phi.dot(b__c_Theta)

## }}}


## c) LS estimate using QR decomp {{{

c__Q, c__R = qr(b__Phi, mode='economic')
c__c_Theta = inv(c__R) .dot(c__Q.T) .dot(a__Y)
c__Y_FIT = b__Phi.dot(c__c_Theta)

## }}}


## d) Recursive LS {{{

# take the samples beforehand
d__X = np.random.uniform(0., 1., pd__n+1)
d__Y = np.array([f(x)+np.random.standard_normal()*pd__var for x in d__X])

# calculate phi_0, Psi_0, z_0
d__x_curr = d__X[0]
d__y_curr = d__Y[0]
d__phi = b__chosen_basis(d__x_curr, pb__d, pb__sigma)
d__Psi = np.outer(d__phi, d__phi)
d__z = d__phi * d__y_curr

# at which iterations do we want to plot
d__plot_at_iter = \
	np.sort([pd__n - i*(pd__n//pd__no_plots) for i in range(pd__no_plots)])
print(f'Plotting at iterations {d__plot_at_iter}')

# save the Yi vectors we want to plot
d__Y_FIT = np.empty(0)

for i, (d__x_curr, d__y_curr) in enumerate(zip(d__X, d__Y)):
	d__phi = b__chosen_basis(d__x_curr, pb__d, pb__sigma)

	d__Psi = d__Psi + np.outer(d__phi, d__phi)
	d__z = d__z + d__phi * d__y_curr

	if (i in d__plot_at_iter):

		# for speed, only calculate c_theta_REC when needed
		if (pd__use_SMF): # Sherman-Morrison Formula
			d__Psi_inv = inv(d__Psi)
			d__SMF = d__Psi_inv - (d__Psi_inv .dot(np.outer(d__phi, d__phi)) .dot(d__Psi_inv)) / \
				(1 + (d__phi .dot(d__Psi_inv) .dot(d__phi)))
			d__c_theta = d__SMF .dot( d__z + d__phi .dot(d__Y[i]) )
		else:
			d__c_theta = inv(d__Psi + np.outer(d__phi, d__phi)) .dot( d__z + d__phi * d__y_curr )

		d__Y_FIT_curr = Phi_gen(np.linspace(0, 1, pd__fitted_points_to_show), \
			b__chosen_basis, pb__d, pb__sigma) .dot(d__c_theta)
		d__Y_FIT = np.append(d__Y_FIT, d__Y_FIT_curr)

## }}}


## e) Least norm {{{

e__X = np.random.uniform(0., 1., pe__n)
e__Y = np.array([f(x)+np.random.standard_normal() for x in e__X])

e__Phi     = Phi_gen(e__X, b__chosen_basis, pb__d, pb__sigma)
e__c_theta = Phi.T .dot(pseudoinverse(Phi.dot(Phi.T))) .dot(e__Y)
#  e__c_theta = e__Phi.T .dot(np.linalg.pinv(e__Phi.dot(e__Phi.T))) .dot(e__Y)
e__Y_FIT   = e__Phi .dot(e__c_theta)

## }}}


## f) Plots and evaluation {{{

s = 3. # marker size
loc = 1 # legend location

# b) and c) subex {{{
# Sample
plt.scatter(a__X, a__Y, label='Sample (y = cxsin(cx) + e)', color='red', s=s)
# LS estimate
plt.scatter(a__X, b__Y_FIT, label=f'LS estimate (using {pb__basis} basis)', color='blue', s=s)
# LS estimate
plt.scatter(a__X, c__Y_FIT, label=f'LS estimate with QR decomp (using {pb__basis} basis)', color='green', s=s)

plt.legend(loc=loc)
plt.grid()
#  plt.savefig('ex_I_1_plots_1.pdf')
#  plt.show()
plt.close()
# }}}

# d) subex {{{
fig, ax = plt.subplots(nrows=pd__no_plots)
for i, j in enumerate(d__plot_at_iter):
	lwb = i*pd__fitted_points_to_show
	upb = (i+1)*pd__fitted_points_to_show
	ax[i].scatter(d__X[:j], d__Y[:j], # Sampled points \
		label=f'Samples for recursive LS ({j})', color='red', s=s)
	ax[i].scatter(np.linspace(0, 1, pd__fitted_points_to_show), d__Y_FIT[lwb:upb], # Fitted points \
		label=f'Recursive LS after {j} samples', color='blue', s=s)
	ax[i].legend(loc=loc)
	ax[i].grid()
#  plt.savefig(f'ex_I_1_plots_2_{i+1}.pdf')
#  plt.show()
plt.close()
# }}}

# e) subex {{{
# Sample
plt.scatter(a__X, a__Y, label='Sample (y = cxsin(cx) + e)', color='red', s=s)
# LS estimate
plt.scatter(a__X, b__Y_FIT, label=f'LN estimate (using {pb__basis} basis)', color='blue', s=s)

plt.legend(loc=loc)
plt.grid()
#  plt.savefig('ex_I_1_plots_3.pdf')
plt.show()
plt.close()
# }}}

## }}}
