import bfgs_simp as bfgs
import numpy as np


def obj_func(x):
    # return np.max(x, 0)
    return np.sum(np.absolute(x))


def obj_grad(x):
    fprime = np.sign(x)
    return fprime


x0 = np.array([10])

H0 = np.identity(1)

x, dx = bfgs.bfgs(obj_func, obj_grad, x0, H0)

print(x, dx)
