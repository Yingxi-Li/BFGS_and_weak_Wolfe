import numpy as np


def line_search(f, fprime, xk, pk):
    """
    Line search enforcing weak Wolfe condition (Armoji and Wolfe condition)
    """
    alpha = 0
    beta = 2**100
    t = 1
    c1 = 1e-4
    c2 = 0.9

    fc = [0]
    gc = [0]

    def fxk1(alpha):
        fc = [0]
        return f(xk + alpha * pk)

    def

    num_iter = 0

    while True:
        if not S(f, fprime, xk, pk, alpha, c1):
            beta = t
        elif not C(f, fprime, xk, pk, alpha, c2):
            alpha = t
        else:
            break

        if beta < 2**100:
            t = (alpha + beta)/2
        else:
            t = 2 * alpha
        num_iter += 1

    return t, num_iter


def S(f, fprime, xk, pk, alpha, c1):
    f_xk = f(xk)
    f_xk1 = f(xk + alpha * pk)
    df_xk = fprime(xk)
    return f_xk1 <= f_xk + c1 * alpha * np.dot(df_xk, pk)


def C(f, fprime, xk, pk, alpha, c2):
    df_xk = fprime(xk)
    df_xk1 = fprime(xk + alpha * pk)
    return -np.dot(pk, df_xk1) <= -c2 * np.dot(pk, df_xk)
