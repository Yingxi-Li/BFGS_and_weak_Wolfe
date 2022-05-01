import bfgs_simp as bfgs
import numpy as np


def obj_func(x):
    # return w|Ax|+(w-1)e1'Ax
    m = 15
    n = 4
    w = 4
    A = np.random.rand(m, n)
    f1 = w*np.linalg.norm(A.dot(x))
    f2 = (w-1)*(A.dot(x))[0]

    return f1+f2


def obj_grad(x):
    # fprime = w*A'Ax/|Ax|+(w-1)A'[:,0]
    fprime = w*A.transpose().dot(A)
    return fprime


x0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]) + \
    np.random.uniform(-0.1, 0.1, size=9)

H0 = np.identity(9)

x, dx, fxs = bfgs.bfgs(obj_func, obj_grad, x0, H0, "weak")

print(fxs)
