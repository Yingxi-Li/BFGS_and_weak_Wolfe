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

    num_iter = 0
    has_found = False
    while num_iter < 100 and has_found == False:

        if not S(f, fprime, xk, pk, t, c1):
            beta = t
        elif not C(fprime, xk, pk, t, c2):
            alpha = t
        else:
            has_found = True

        if has_found == True:
            break

        if beta < 2**100:
            t = (alpha + beta)/2
        else:
            t = 2 * alpha
        num_iter += 1

    print("line search iterations:", num_iter)
    # return t, num_iter
    return t


def S(f, fprime, xk, pk, t, c1):
    f_xk = f(xk)
    f_xk1 = f(xk + t * pk)
    df_xk = fprime(xk)
    # print(f_xk1, f_xk)
    return f_xk1 < f_xk + c1 * t * np.dot(df_xk, pk)  # s = gradientf dot pk


def C(fprime, xk, pk, t, c2):
    df_xk = fprime(xk)
    df_xk1 = fprime(xk + t * pk)
    return -np.dot(pk, df_xk1) < -c2 * np.dot(pk, df_xk)


def C2(fprime, xk, pk, t, c2):
    df_xk = fprime(xk)
    df_xk1 = fprime(xk + t * pk)
    hprimet = np.dot(pk, df_xk1)
    s = np.dot(pk, df_xk)
    hprimet > c2 * s
