import numpy as np
from numpy.linalg import inv, det
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sympy as sy
from SVD import pseudoinverse
from ex_2_param import *


## Utility {{{

def ellipse_transform(A, B, C, q):
	a, b, alpha = sy.symbols('a, b, alpha')

	# A * x**2 + 2*B * x*y + C * y**2 = q

	eq1 = sy.cos(alpha)**2 / a**2 + sy.sin(alpha)**2 / b**2 - A/q
	eq2 = sy.sin(alpha) * sy.cos(alpha) * (1/a**2 - 1/b**2) - B/q
	eq3 = sy.sin(alpha)**2 / a**2 + sy.cos(alpha)**2 / b**2 - C/q

	sol = sy.nsolve((eq1, eq2, eq3), (a, b, alpha), (.007, .01, np.radians(45)), prec=3)
	assert(sol)
	a, b, alpha = (sol[0], sol[1], sol[2])
	return a, b, alpha

## }}}


## Sample generation {{{

def fun(y0, y1, n, a, b, var=1):
	ret = np.empty(n)
	ret[:2] = np.array([y0, y1])
	for i in range(2, n):
		ret[i] = a*ret[i-1] + b*ret[i-2] + np.random.standard_normal()*var
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

## }}}


## Covariance estimation {{{

Gamma_n = Phi.T @ Phi / p__n
print(f'Covariance:\n {Gamma_n}')

## }}}


## Confidence ellipsoids {{{

df = 2 # degrees of freedom
p_i = np.array([.9, .95, .98])
q_i = np.array([chi2.cdf(p, df=df) for p in p_i])
print(f'p_i: {p_i} -> q_i = {q_i}')

#  sy_plots = None
ellipses = []
ellipses_2 = []
colo=['r', 'b', 'g']
for i, (q, c) in enumerate(zip(q_i, colo)):
	A = Gamma_n[0, 0]
	B = 2*Gamma_n[0, 1]
	C = Gamma_n[1, 1]

	ellipses.append((A, B, C, q, c))

	#  a, b, alpha = ellipse_transform(A, B, C, q/p__n)
	#  width = 2*a
	#  height = 2*b
	#  print(f'Ellipse{i}: {a}, {b}, {np.degrees(float(alpha))}')
	#  ellipses_2.append(Ellipse((h_theta[0], h_theta[1]), width, height, alpha, \
	#      fill=False, color=c, label=f'$P={p_i[i]*100:.0f}\%$'))

## }}}


## Plots and evaluation {{{

s = 3. # marker size
loc = 1 # legend location

# Sample
plt.scatter(T[:p__n], Y[:p__n], label='Sample ($y[n] = a\\,y[n-1] + b\\,y[n-2] + \\epsilon$)', color='red', s=s)
# LS
plt.plot(T[:p__n], s_Y_LS[:p__n], label='LS with $\\theta=\\theta^*$', color='green')
# LS
plt.plot(T[:p__n], Y_LS[:p__n], label='LS with $\\theta=\\hat{\\theta}$', color='blue')

plt.legend(loc=loc)
plt.xlabel('$\\Delta t$ Time step')
plt.ylabel('$y_t$ values')
plt.grid()
#  plt.savefig('../figures/ex_I_2_plots_1.pdf')
plt.show()
plt.close()

plt.scatter([s_theta[0]], [s_theta[1]], color='black')
ax  = plt.gca()

for i, (A, B, C, q, color) in enumerate(ellipses):
	bound = np.sqrt(q / (A - B**2/(4*C)))

	no_points = 60
	X = np.linspace(-bound, bound, no_points)
	Y_p = np.empty((no_points,))
	Y_n = np.empty((no_points,))

	for j, x in enumerate(X):
		a = 1
		b = B/C * x
		c = (A*x**2 - q) / C
		Y_p[j] = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
		Y_n[j] = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a)

	ax.plot(X+h_theta[0], Y_p+h_theta[1], color=color, label=f'$P={p_i[i]*100:.0f}\%$')
	ax.plot(X+h_theta[0], Y_n+h_theta[1], color=color)

#  for e in ellipses_2:
#      ax.add_patch(e)

plt.xlabel('$\\hat{\\theta}_1$')
plt.ylabel('$\\hat{\\theta}_2$')
plt.legend()
plt.grid()
#  plt.savefig('../figures/ex_I_2_plots_2.pdf')
plt.show()

## }}}
