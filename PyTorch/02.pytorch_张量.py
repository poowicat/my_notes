from os import X_OK
import torch
from torch._C import device
from torch.functional import Tensor
import numpy as np
# 創建一个5x3矩阵，但是未初始化
x = torch.empty(5, 3)
print(x)
# 创建一个随机初始化的矩阵
x = torch.rand(5, 3)
print(x)

# 创建一个0填充的矩阵，数据类型为long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 创建tensor并使用现有数据初始化
x = torch.tensor([5.5, 3])
print(x)  # tensor([5.5000, 3.0000])

# 根据现有的张量创建张量，这些方法将重用输入张量的属性，例如，dtype，除非设置新的值进行覆盖
x = x.new_ones(5, 3, dtype=torch.double)  # 用new_*方法来创建对象
print(x)

x = torch.randn_like(x, dtype=torch.float)  # 覆盖 dtype！
print(x)
print(x.size)  # 使用size方法与Numpy的shape属性返回的相同，张量也支持shape属性

torch.Size([5, 3])

# 加法操作
#  加法1
y = torch.rand(5, 3)
print(x+y)

# 加法2
print(torch.add(x, y))


# 提供输出tensor作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# 替换
# adds x to y
# 任何 以``_`` 结尾的操作都会用结果替换原变量. 例如: ``x.copy_(y)``, ``x.t_()``, 都会改变 ``x``.
y.add_(x)
print(y)


# 可以使用numpy索引方式相同的操作来进行对张量的操作
print(x[:, 1])


# torch.view 与 Numpy的reshape类似，可以改变张量的维度大小
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # size -1 从其他维度推断
# torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
print(x.size(), y.size(), z.size())

# 若你只有一个元素的张量，使用.item()来得到Python数据类型的数值
x = torch.randn(1)
print(x)
print(x.item())

"""Numpy转换"""
# 将一个Torch Tensor转换为Numpy数组是一件轻松的事，反之亦然
# Torch Tensor 与 Numpy 数组共享底层内存地址，修改一个对导致另一个变化

# 将一个tensor转换成Numpy数组
a = torch.ones(5)
print(a)  # tensor([1., 1., 1., 1., 1.])
b = a.numpy()
print(b)  # [1. 1. 1. 1. 1.]

# 观察 numpy 数组的值是如何改变的
a.add_(1)
print(a)  # tensor([2., 2., 2., 2., 2.])
print(b)  # [2. 2. 2. 2. 2.]

# Numpy Array 转化成 Torch Tensor
# 使用from_numpy自动转化
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)  # [2. 2. 2. 2. 2.]
print(b)  # tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
# 所有的tensor类型默认都是基于CPU，CharTensor类型不支持到Numpy的转换

"""CUDA张量"""
# 使用.to方法 可以将Tensor 移动到任何设备中

# is_available 函数判断是否有cuda可以使用
# ''torch.device''将张量移动到指定的设备中

if torch.cuda.is_available():
    device = torch.device("cuda")  # coda设备对象
    y = torch.ones_like(x, device=device)  # 直接从CPU创建张量
    x = x.to(device)  # 或者直接使用".to("cuda")" 将张量移动到cuda中
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))  # ".to" 也会对变量的类型做更改

# 1.张量的索引和切片
tensor = torch.ones(4, 4)
tensor[:, 1] = 0  # 将第1列(从0开始)的数据全部赋值为0
print(tensor)
"""tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])"""
# 2.张量的拼接
"""你可以用torch. cat方法将一组张量按照指定的维度进行拼接，也可参考torch. stack方法。这个方法也可实现拼接操作，和cat稍有不同"""
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 3.张量的乘积和矩阵乘法

# 逐个相乘结果（乘积）
print(f"tensor.mul(tensor:\n{tensor.mul(tensor)} \n")
# 等价写法：
print(f"tensor * tensor:\n{tensor * tensor}")
# 下面写法表示张量与张量的矩阵乘法：
print(tensor.T)
print(f"tensor.matmul(tensor.T):\n {tensor.matmul(tensor.T)} \n")
# 等价写法：
print(f"tensor @ tensor.T: \n {tensor @ tensor.T}")


# 4.自动赋值运算
"""自动赋值运算通常在方法后有_作为后缀，例如x.copy_(y), x.t_() 操作都会改变 x 的取值"""
print(tensor, '\n')
tensor.add_(5)  # 注意：自动赋值可以节省内存，但在求导时会出现问题，所以不鼓励使用它
print(tensor)

