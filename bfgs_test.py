import bfgs_simp as bfgs
import numpy as np


def obj_func(x):
    # return np.max(x, 0)
    return 8*np.linalg.norm(x)+7*x[0]


def obj_grad(x):
    fprime = np.array([8*x[0]/np.linalg.norm(x)+7, 8*x[1]/np.linalg.norm(x)])
    return fprime


x0 = np.array([10, 10])

H0 = np.identity(2)

x, dx = bfgs.bfgs(obj_func, obj_grad, x0, H0)

print(x, dx)
