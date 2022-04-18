import line_search_strong as strong
import line_search_weak as weak
import numpy as np


def obj_func(x):
    return 8*np.linalg.norm(x)+7*x[0]


def obj_grad(x):
    return np.array([8*x[0]/np.linalg.norm(x)+7, 8*x[1]/np.linalg.norm(x)])


start_point = np.array([5, 5])
search_gradient = np.array([-1.0, -1.0])


print(weak.line_search(
    obj_func, obj_grad, start_point, search_gradient))
