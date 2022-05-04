import numpy as np
import line_search_weak as weak
import line_search_strong as strong


def bfgs(f, fprime, x0, H0, ls_type):

    dim = x0.shape[0]

    k = 0
    Hk = H0
    xk = x0
    fxs = []

    ls_iters = []

    is_weak = (ls_type == "weak")

    while k <= 500:
        fxs.append(f(xk))
        # print("iteration:", k)
        # print("iteration:", k, "x:", xk, "fx", f(xk))
        # print("f'(xk)", fprime(xk))
        pk = -Hk @ fprime(xk).T
        # print("Hk", Hk)
        # print("eigs", np.linalg.eig(Hk))
        print("xk", xk)
        print("pk", pk)
        if is_weak:
            t, ls_iter = weak.line_search(f, fprime, xk, pk)
            ls_iters.append(ls_iter)
        else:
            t, ls_iter = strong.line_search(f, fprime, xk, pk)
            ls_iters.append(ls_iter)
        print("t", t)
        dfxk = fprime(xk)
        xk = xk + t * pk
        dfxk1 = fprime(xk)

        yk = np.array(dfxk1) - np.array(dfxk)
        # print("yk", yk)
        # if (np.all((yk == 0))):
        #     print('yk is 0!')
        #     yk = yk + 1e-2 * np.random.rand(dim)

        # print('fxk1', f(xk), 'fxk', fx)
        if np.all(yk == 0):  # compare zero lists
            break

        Hk = update(pk, yk, Hk, t, dim)
        k = k + 1

    return xk, f(xk), fxs, ls_iters


def update(pk, yk, Hk, t, dim):
    # print("yk", yk)
    # print("yk.pk", np.dot(yk, pk))
    Vk = np.identity(dim) - np.outer(pk, yk) / np.dot(yk, pk)
    Hk1 = Vk @ Hk @ Vk.T + t / np.dot(yk, pk) * np.outer(pk, pk)
    return Hk1
    # return Hk1 + 1e-12 * np.identity(dim)
