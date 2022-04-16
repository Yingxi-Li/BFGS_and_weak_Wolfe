import numpy as np
import line_search_weak as weak


def bfgs(f, fprime, x0, H0):

    dim = x0.shape

    k = 0
    Hk = H0
    xk = x0

    while k <= 20:
        print(k, xk)

        pk = -Hk * fprime(x0)
        alpha = weak.line_search(f, fprime, xk, pk)

        dfxk = fprime(xk)
        xk = xk + alpha * pk
        dfxk1 = fprime(xk)
        if dfxk1 == 0:
            break
        yk = dfxk1 - dfxk

        Hk = update(pk, yk, Hk, alpha, dim)

        k = k + 1

    return xk, yk,


def update(pk, yk, Hk, alpha, dim):
    Vk = np.identity(dim) - pk @ yk.T / np.dot(yk, pk)
    Hk1 = Vk @ Hk @ Vk.T + alpha * np.dot(yk, pk) * pk @ pk.T
    return Hk1
