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


# # Lasso
# def f(x):
#     return np.absolute(x[0]) + x[1]**2

# def df(x):
#     return np.array([np.sign(x[0]), 2*x[1]])


# # Lasso
n = 8
# A = np.random.rand(n, n) + 10 * np.identity(n)
eig = np.zeros(n)

for i in range(n):
    eig[i] = 0.8 ** i

S = np.random.rand(n, n)
S_inv = np.linalg.inv(S)

A = S @ np.diag(eig) @ S_inv

# eig = np.full(n, np.random.rand(),)
# S = np.random.rand(n, n)
# S_inv = np.linalg.inv(S)

# A = S @ np.diag(eig) @ S_inv


def f(x):
    return np.linalg.norm(np.matmul(A, x))**2 + np.linalg.norm(x, ord=1)


def df(x):
    return 2 * np.matmul(A, x) + np.sign(x)


# results = []
# num_iters = []

# for i in range(50):
#     x0_1 = np.random.uniform(-2, 4)
#     x0 = np.array([x0_1, 20 - np.absolute(x0_1)])

#     H0 = np.identity(2)

#     x, dx, fxs, num_iter = bfgs.bfgs(f, df, x0, H0, "strong")

#     num_iters.append(num_iter)

#     results.append(dx)

# # print(x0)
# # print(fxs)
# print(results)
# print(num_iters)


x0 = 10 * np.random.uniform(-1, 1, n)

H0 = np.identity(n)

x, dx, fxs, num_iter = bfgs.bfgs(f, df, x0, H0, "weak")

print(fxs)
