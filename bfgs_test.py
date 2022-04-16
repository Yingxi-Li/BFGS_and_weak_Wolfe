import bfgs_simp as bfgs
import numpy as np


def obj_func(x):
    return (x[0])**2+(x[1])**2


def obj_grad(x):
    return np.array([2*x[0], 2*x[1]])


x0 = np.array([10, 10])

H0 = np.identity(2)

x, dx = bfgs.bfgs(obj_func, obj_grad, x0, H0)

print(x, dx)
