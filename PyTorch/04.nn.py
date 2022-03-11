'''
Author: your name
Date: 2022-01-07 15:38:39
LastEditTime: 2022-01-10 14:07:11
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: \PyTorch\04.nn.py
'''
import torch.optim as optiom
import torch
import torch.nn as nn
import torch.nn.functional as F

"""定义网络"""


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 2 input image channel , 6 output channels ,5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation : y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """只需要定义forward函数，即可以使用autograd为你自动定义backward函数（计算梯度），可以在forward函数中使用任何张量操作"""
        # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)


# 模型的科学习参数由net.parameters()返回
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
"""输出
10
torch.Size([6, 1, 5, 5])
"""

input = torch.randn(1, 1, 32, 32)  # 定义张量
out = net(input)  # 在forward函数中操作
# tensor([[ 0.0113, -0.0156, -0.0581,  0.0191, -0.0068,  0.0167, -0.0323, -0.0384,0.1187,  0.1024]], grad_fn=<AddmmBackward0>)
print(out)


# 使用随机梯度将所有参数和反向传播的梯度缓冲区归零：
net.zero_grad()
out.backward(torch.randn(1, 10))


"""损失函数"""
# 均分误差的计算
output = net(input)
target = torch.randn(10)  # a dummy target , for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)  # tensor(0.6926, grad_fn=<MseLossBackward0>)

#  现在，如果使用.grad_fn属性向后跟随loss，您将看到一个计算图，如下所示：
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> view -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # Relu
# <MseLossBackward0 object at 0x000002515DFBC820>
# <AddmmBackward0 object at 0x000002515DFBC5B0>
# <AccumulateGrad object at 0x000002515DFBC5B0>


"""反向传播"""
net.zero_grad()  # zeros the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad.after backward')
print(net.conv1.bias.grad)
"""
conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad.after backward
tensor([ 0.0164,  0.0134, -0.0294, -0.0043,  0.0072,  0.0091])
"""
# 以上我们已经看到了如何使用损失函数

"""更新权重"""
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# 可构建一个小包装： torch.optim, 可实现所有这些方法
# Create your optimizer(优化器)
optimizer = optiom.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()  # zero the gradient buffers (0的坡度缓冲区)
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # Does the update




