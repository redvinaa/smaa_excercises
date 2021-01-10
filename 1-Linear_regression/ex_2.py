import numpy as np
from numpy.linalg import inv, det
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from SVD import pseudoinverse
from ex_2_param import *


## Sample generation {{{

def fun(y0, y1, n, a, b, var=1):
	ret = np.empty(n)
	ret[:2] = np.array([y0, y1])
	for i in range(2, n):
		ret[i] = a*ret[i-1] + b*ret[i-2] + np.random.standard_normal() * var
	return ret

T = np.linspace(0, 99, p__n+2) # time steps
Y = fun(p__y0, p__y1, p__n+2, p__a, p__b)
# because of phi_t = [y_t-1, y_t-2], 
# for Phi to be of shape n x 2, we need a
# sample size of n + 2

## }}}


## LS solution {{{

# s means ^* / star
s_theta = np.array([p__a, p__b])
s_Y_LS = fun(p__y0, p__y1, p__n, p__a, p__b, var=0)

Phi = np.empty(shape=(p__n, 2)) # Phi.shape = n x 2
for i in range(p__n):
	Phi[i, :] = np.array([Y[i+1], Y[i]])

h_theta = inv(Phi.T @ Phi) @ Phi.T @ Y[2:]
print(f'h_theta = {h_theta}')

# Y_LS is \hat{\theta}_n
Y_LS = np.empty(p__n)
Y_LS[:2] = np.array([p__y0, p__y1])
for i in range(2, p__n):
	Y_LS[i] = np.array([Y_LS[i-1], Y_LS[i-2]]) @ h_theta
enumerate(zip(Phi, Y))
## }}}


## Covariance estimation {{{

Gamma_n = Phi.T @ Phi / p__n
print(f'Covariance:\n {Gamma_n}')

## }}}


## Confidence ellipsoids {{{

df = 2 # degrees of freedom
p_i = np.array([.9, .95, .98])
q_i = np.array([chi2.ppf(p, df=df) for p in p_i])
print(f'p_i: {p_i} -> q_i = {q_i}')

ellipses = []
colo=['r', 'b', 'g']
for i, (q, c) in enumerate(zip(q_i, colo)):
	A = Gamma_n[0, 0]
	B = 2*Gamma_n[0, 1]
	C = Gamma_n[1, 1]

	# eq. of ellipse:    A   * x**2  +  2 *   B   * x*y  +    C   * y**2 = q
	# eq. of ellipse:  (A/q) * x**2  +  2 * (B/q) * x*y  +  (C/q) * y**2 = 1
	ellipses.append((A/q, B/q, C/q, c))

## }}}


## Plots and evaluation {{{

s = 3. # marker size
loc = 1 # legend location

for i in range(p__n-2):
	pt1 = Y[i]
	pt2 = Y[i+1]
	pt_pred = np.array([pt1, pt2]) @ h_theta
	pt_opt  = np.array([pt1, pt2]) @ s_theta
	if i == 0:
		plt.plot(T[i:i+2], np.array([pt1, pt2]), color='red', label='data')
		plt.plot(T[i+1:i+3], np.array([pt2, pt_pred]), color='blue', label='predicted')
		plt.plot(T[i+1:i+3], np.array([pt2, pt_opt]),  color='green', label='optimal')
	else:
		plt.plot(T[i:i+2], np.array([pt1, pt2]), color='red')
		plt.plot(T[i+1:i+3], np.array([pt2, pt_pred]), color='blue')
		plt.plot(T[i+1:i+3], np.array([pt2, pt_opt]),  color='green')

plt.legend(loc=loc)
plt.xlabel('$\\Delta t$ Time step')
plt.ylabel('$y_t$ values')
plt.grid()
plt.savefig('../figures/ex_I_2_plots_1.pdf')
plt.show()
plt.close()

plt.scatter([s_theta[0]], [s_theta[1]], color='black',  marker='x', label='$\\theta^*$')
plt.scatter([h_theta[0]], [h_theta[1]], color='purple', marker='x', label='$\\hat{\\theta}$')
ax  = plt.gca()

for i, (A, B, C, color) in enumerate(ellipses):
	#  x bound of ellipse
	bound = np.sqrt(4*C / (4*A*C - B**2))

	no_points = 10000
	X = np.linspace(-bound, bound, no_points)
	Y_p = np.empty((no_points,))
	Y_n = np.empty((no_points,))

	for j, x in enumerate(X):
		a = C
		b = B * x
		c = A*x**2 - 1
		Y_p[j] = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
		Y_n[j] = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

	ax.plot(X+h_theta[0], Y_p+h_theta[1], color=color, label=f'$P={p_i[i]*100:.0f}\%$')
	ax.plot(X+h_theta[0], Y_n+h_theta[1], color=color)

plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.legend()
plt.grid()
plt.savefig('../figures/ex_I_2_plots_2.pdf')
plt.show()

## }}}
