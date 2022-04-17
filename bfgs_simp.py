import numpy as np
import line_search_weak as weak


def bfgs(f, fprime, x0, H0):

    dim = x0.shape[0]

    k = 0
    Hk = H0
    xk = x0

    while k <= 10:
        print("iteration:", k, "x:", xk)
        print("Hk", Hk)
        pk = -Hk @ fprime(x0)
        print("pk:", pk)
        t = weak.line_search(f, fprime, xk, pk)
        print("t", t)
        fx = f(xk)
        dfxk = fprime(xk)
        xk = xk + t * pk
        dfxk1 = fprime(xk)

        yk = np.array(dfxk1) - np.array(dfxk)

        if f(xk) < 1e-3:  # compare zero lists
            break

        Hk = update(pk, yk, Hk, t, dim)

        k = k + 1

    return xk, f(xk)


def update(pk, yk, Hk, t, dim):
    print("yk", yk)
    Vk = np.identity(dim) - pk @ yk.T / np.dot(yk, pk)
    Hk1 = Vk @ Hk @ Vk.T + t / np.dot(yk, pk) * pk @ pk.T
    return Hk1
