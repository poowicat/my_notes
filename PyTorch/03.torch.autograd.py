import torchvision,torch

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# 正向传播
prediction = model(data)

# 反向传播
loss = (prediction - labels).sum()
loss.backward()  # backward pass

# 加载优化器 SGD 学习率为0.01，动量为0.9
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# 调用.step()启动梯度下降，优化器通过.grad中存储的梯度来调整每个参数
optim.step()  # gradient descent

# 我们从a和b创建另一个张量Q。
# 假设a、b是神经网络参数，Q是误差。在NN训练中，我们想要对于参数的误差即当在Q上
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)


# 调用.backward()时，autograd将计算这些梯度并将其存储在各个张量的grad属性中
# 我们需要在Q.backward()中显式传递gradient参数，因为它是向量，gradient是与Q形状相同的张量，它表示Q相对于本身的梯度，即

Q = 3*a**3 - b**2

# 同样我们也可以将Q
# 聚合为一个标量，然后隐式地向后调用，例如 Q.sum().backward()
external_grad = torch.tensor([1,1])
Q.backward(gradient=external_grad)

# 梯度线程沉积在a.grad 和 b.grad中

# check if collected gradients are corrent
print(9*a**2 == a.grad)
print(-2*b == b.grad)
"""tensor([True, True])
tensor([True, True])
"""



