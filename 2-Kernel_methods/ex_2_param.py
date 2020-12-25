# file: I_2/ex_II_2_parameters.py

from numpy import array

p__n = 100
p__R = 3

p__cov_a = array([[1, 0], [0, 1]])
p__cov_b = .3

p__eps = .1
p__lambd = 1

#  p__mode = 'gaussian'
#  p__params = {'sigma': 1}
p__mode = 'linear'
p__params = {}
