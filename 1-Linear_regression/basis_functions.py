import numpy as np

def poly_basis(x, d, sigma):
	# seems to be the best for the given function
	# works well with d>=10
	phi = np.empty(shape=(d,))
	for i in range(d):
		mu_i = i
		phi[i] = x**mu_i
	return phi

def sin_basis(x, d, sigma):
	# works well with d=10 and sigma=0.5
	phi = np.empty(shape=(d,))
	for i in range(d):
		mu_i = i+1
		phi[i] = sigma * np.sin(mu_i*x)
	return phi

def exp_basis(x, d, sigma):
	# works well with d=10 and sigma=0.5
	phi = np.empty(shape=(d,))
	for i in range(d):
		mu_i = i / (d+1)
		phi[i] = np.exp( - np.abs(x-mu_i)**2 / sigma**2 )
	return phi
