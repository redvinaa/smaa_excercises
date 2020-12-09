#! /usr/bin/env python3
# file: I_1/ex_I_1.py

import numpy as np
from numpy.linalg import inv
from scipy.linalg import qr
import matplotlib.pyplot as plt
from ex_I_1_parameters import *


## a) Sample generation {{{

def f(x):
	return c * x * np.sin(c * x)

X = np.random.uniform(0., 1., n)
Y = np.array([f(x)+np.random.standard_normal() for x in X])

## }}}


## b) Linear regression estimate {{{

from basis_functions import poly_basis, sin_basis, exp_basis

def Phi_gen(X, basis, d, sigma):
	n = X.size
	Phi = np.empty(shape=(n,d,))
	for i in range(n):
		Phi[i,:] = basis(X[i], d, sigma)
	return Phi

chosen_basis = poly_basis

Phi = Phi_gen(X, chosen_basis, d, sigma)

# c_ means cap/optimal
c_Theta_LS = inv(Phi.T .dot(Phi)) .dot(Phi.T) .dot(Y)
Y_LS = Phi.dot(c_Theta_LS)

## }}}


## c) LS estimate using QR decomp {{{

Q, R = qr(Phi, mode='economic')
c_Theta_QR = inv(R) .dot(Q.T) .dot(Y)
Y_QR = Phi.dot(c_Theta_QR)

## }}}


## d) Recursive LS {{{

## }}}


## e) Least norm {{{

## }}}


## f) Plots and evaluation {{{

s = 1. # marker size

# Sample
plt.scatter(X, Y, label='Sample', color='red', s=s)
# LS estimate
plt.scatter(X, Y_LS, label='LS estimate (using sin basis)', color='blue', s=s)
# LS estimate
plt.scatter(X, Y_QR, label='LS estimate with QR decomp (using poly basis)', color='green', s=s)

plt.legend()
plt.grid()
plt.show()

## }}}
