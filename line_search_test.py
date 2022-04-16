import line_search_strong as strong
import line_search_weak as weak
import numpy as np


def obj_func(x):
    return (x[0])**2+(x[1])**2


def obj_grad(x):
    return [2*x[0], 2*x[1]]


start_point = np.array([10, 10])
search_gradient = np.array([-1.0, -1.0])

# print(strong.line_search(
#     obj_func, obj_grad, start_point, search_gradient))

print(weak.line_search(
    obj_func, obj_grad, start_point, search_gradient))
