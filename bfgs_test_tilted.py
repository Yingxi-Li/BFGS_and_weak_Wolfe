import bfgs_simp as bfgs
import numpy as np

w = 4
m = 15
n = 500
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


# x0 = np.array([-11, 9, 10, -8, -2, 5, -7, 1])
# results = []
# xs = []
# ys = []
# for i in range(1000):
#     x0_1 = np.random.uniform(-10, 10)
#     x0_2 = np.random.uniform(-10, 10)
#     x0 = np.array([x0_1, x0_2])

#     H0 = np.identity(n)

#     x, dx, fxs, ls_iters = bfgs.bfgs(obj_func, obj_grad, x0, H0, "weak")

#     results.append(dx)
#     xs.append(x0_1)
#     ys.append(x0_2)

# print("results:", results)
# print("xs", xs)
# print("yx", ys)

x0 = 10 * np.random.uniform(-1, 1, n)

H0 = np.identity(n)

x, dx, fxs, ls_iters = bfgs.bfgs(obj_func, obj_grad, x0, H0, "weak")

print(fxs)
