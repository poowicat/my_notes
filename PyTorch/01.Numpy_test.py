import numpy as np
# import minpy
from minpy.core import grad
# from __future__ import print_function
import torch

"""创建numpy数组的方式之——从Python列表装换"""
array = np.array([4, 5, 6])
print(array)  # [4,5,6]

"""使用特殊库函数"""
a = np.random.random((2, 2))
print(a)


def foo(x):
    if x >= 0:
        return x
    else:
        return 2 * x


foo_grad = grad(foo)
print(foo_grad(3))  # should print 1.0
print(foo_grad(-1))  # should print 2.0

x = torch.empty(5, 3)
print(x)
