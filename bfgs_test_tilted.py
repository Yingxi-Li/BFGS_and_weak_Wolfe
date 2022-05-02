import bfgs_simp as bfgs
import numpy as np

w = 4
m = 15
n = 2
A = np.random.rand(m, n)


def obj_func(x):
    # # return f(x)=w|Ax|+(w-1)e1*A*x
    # f1 = w*np.linalg.norm(A.dot(x))
    # f2 = (w-1)*((A.dot(x))[0])
    # y = f1+f2

    f1 = w*np.linalg.norm(x)
    y = f1 + (w - 1)*x[0]

    return y


def obj_grad(x):
    # # return w*A'Ax/|Ax|+(w-1)A'[:,0]
    # g1 = w*A.transpose().dot(A.dot(x))
    # g2 = (w-1)*(A.T)[:, 0]
    # fprime = g1+g2

    g1 = w * x/np.linalg.norm(x)
    fprime = g1

    return fprime


x0 = np.array([11, 9])

H0 = np.identity(n)

x, dx, fxs = bfgs.bfgs(obj_func, obj_grad, x0, H0, "weak")

print(fxs)
