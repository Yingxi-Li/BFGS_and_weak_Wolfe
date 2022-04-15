import numpy as np


def line_search():
    """
    Line search enforcing weak Wolfe condition (Armoji and Wolfe condition)
    """
    alpha = 0
    beta = 2**100
    t = 1

    while True:
        if not S(t):
            beta = t
        elif not C(t):
            alpha = t
        else:
            break

        if beta < 2**100:
            t = (alpha + beta)/2
        else:
            t = 2 * alpha

# To implement the two conditions, check c1, c2 used in scipy source code


def S(t):
    return True


def C(t):
    return True
