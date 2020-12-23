# file: I_1/ex_I_1_parameters.py

from numpy import array

p__lambd = 1

p__n        = 100                 # Sample size
p__m_a      = []
p__m_b      = []
p__cov_a    = []
p__cov_b    = []

p__m_a  .append([3, 0])               # Mean of 1a
p__m_b  .append([0, 0])               # Mean of 1b
p__cov_a.append([[1, 0], [0, 1]])     # Covariance matrix of 1a
p__cov_b.append([[1, 0], [0, 1]])     # Covariance matrix of 1b

p__m_a  .append([2, 1])               # Mean of 2a
p__m_b  .append([-1, -2])             # Mean of 2b
p__cov_a.append([[1, .5], [.5, 1]])   # Covariance matrix of 2a
p__cov_b.append([[1, -.2], [-.2, 1]]) # Covariance matrix of 2b

p__m_a  .append([5, 5])               # Mean of 3a
p__m_b  .append([-3, -3])             # Mean of 3b
p__cov_a.append([[1, 0], [0, 1]])     # Covariance matrix of 3a
p__cov_b.append([[1, 0], [0, 1]])     # Covariance matrix of 3b
