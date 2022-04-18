import bfgs_simp as bfgs
import numpy as np


def obj_func(x):
    # return 8*np.linalg.norm(x)+7*x[0]
    return np.sum(np.absolute(x))


def obj_grad(x):
    # fprime = np.array([8*x[0]/np.linalg.norm(x)+7, 8*x[1]/np.linalg.norm(x)])
    fprime = np.sign(x)
    return fprime


x0 = np.array([10, 10])

H0 = np.identity(2)

x, dx = bfgs.bfgs(obj_func, obj_grad, x0, H0, "weak")

print(x, dx)
