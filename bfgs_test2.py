import bfgs_simp as bfgs
import numpy as np

# piece wise linear
num_pieces = 10
n7 = 2

np.random.seed(1)
A = np.random.uniform(-10, 10, size=[num_pieces, n7])
b = np.random.uniform(0, 5, num_pieces)


def f(x): return np.max([(A[i] @ x.T + b[i]) for i in range(num_pieces)])
def df(x): return A[np.argmax([A[i] @ x.T + b[i] for i in range(num_pieces)])]


x0 = np.array([10, -8])

H0 = np.identity(2)

x, dx = bfgs.bfgs(f, df, x0, H0)

print(x, dx)
