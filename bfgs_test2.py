import bfgs_simp as bfgs
import numpy as np

# # piece wise linear
# num_pieces = 10
# n7 = 2

# np.random.seed(1)
# A = np.random.uniform(-10, 10, size=[num_pieces, n7])
# b = np.random.uniform(0, 5, num_pieces)


# def f(x): return np.max([(A[i] @ x.T + b[i]) for i in range(num_pieces)])
# def df(x): return A[np.argmax([A[i] @ x.T + b[i] for i in range(num_pieces)])]


# # l1-norm
# def f(x): return np.sum(np.absolute(x))


# def df(x): return np.sign(x)

# Lasso
def f(x):
    return np.absolute(x[0]) + x[1]**2


def df(x):
    return np.array([np.sign(x[0]), 2*x[1]])


x0 = np.array([19, 1])

H0 = np.identity(2)

x, dx, fxs = bfgs.bfgs(f, df, x0, H0, "weak")

print(fxs)
