import bfgs_simp as bfgs
import numpy as np


def obj_func(x):
    # return 8*np.linalg.norm(x)+7*x[0]
    return np.sum(np.absolute(x))


def obj_grad(x):
    # fprime = np.array([8*x[0]/np.linalg.norm(x)+7, 8*x[1]/np.linalg.norm(x)])
    fprime = np.sign(x)
    return fprime


n = 100

# x0 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1]) + \
#     np.random.uniform(-0.1, 0.1, size=9)
# x0 = 10 * np.ones(n) + np.random.rand(n)
# x0_1 = np.random.uniform(-10, 10)
# x0 = np.array([x0_1, 10-np.absolute(x0_1)])
x0 = 10 * np.random.uniform(-1, 1, size=n)

H0 = np.identity(n)

x, dx, fxs, iters = bfgs.bfgs(obj_func, obj_grad, x0, H0, "strong")
print(x0)
print(fxs)
