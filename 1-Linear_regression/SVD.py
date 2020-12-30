# file: I_1/SVD.py

import numpy as np
from numpy.linalg import inv, eig

def SVD(Phi):
	eig_val, U = eig(Phi.dot(Phi.T))
	eig_val, V = eig(Phi.T.dot(Phi))
	S = np.zeros_like(Phi)
	for i, val in enumerate(eig_val):
		S[i,i] = val

	# Phi.shape = n x m
	# U.shape   = n x n
	# S.shape   = n x m
	# V.shape   = m x m
	return (U, S, V)

def pseudoinverse(Phi):
	U, S, V = SVD(Phi)

	# Moore-Penrose pseudoinverse of S
	S_inv = inv(S.T @ S) @ S.T
	return V * S_inv * U.T
