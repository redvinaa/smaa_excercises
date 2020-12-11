# file: I_1/SVD.py

import numpy as np
from scipy.linalg import inv

def SVD(Phi):
	U     = np.array([])
	Sigma = np.array([])
	V     = np.array([])
	return (U, Sigma, V)

def pseudoinverse(Phi):
	U, Sigma, V = SVD(Phi)
	return V*inv(Sigma)*U.T
