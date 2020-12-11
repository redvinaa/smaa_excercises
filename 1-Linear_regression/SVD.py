# file: I_1/SVD.py

import numpy as np
from scipy.linalg import inv

def SVD(Phi):
	eig_val, U = np.linalg.eig(Phi.dot(Phi.T))
	eig_val, V = np.linalg.eig(Phi.T.dot(Phi))
	S = np.zeros_like(Phi)
	for i, val in enumerate(eig_val):
		S[i,i] = val
	return (U, S, V)

def pseudoinverse(Phi):
	U, S, V = SVD(Phi)
	return V*inv(S)*U.T
