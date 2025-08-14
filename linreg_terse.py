import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=d2l.synthetic_data(true_w,true_b,1000)

#不需要进行批量大小和是否打乱，直接传入数据集 告诉批量大小batch_size,再告诉它需要打乱就会实现打乱批量喂取数据
def load_array(data_arrays,batch_size,is_train=True):
    """构造一个pytorch的数据迭代器"""
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
batch_size=10
data_iter=load_array((features,labels),batch_size)
next(iter(data_iter))

#调用了线性回归函数
from torch import nn
net=nn.Sequential(nn.Linear(2,1))

#旧版需要为w和b分配内存初始化值，现在nn.Linear已经创建好了e和b，看可以直接访问直接初始化
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

#损失函数也隐藏起来直接调用了
loss=nn.MSELoss()

#这里就是随机梯度下降优化器了，会自动获取net中所有需要优化的参数w和b，还定义了学习率0.03
trainer=torch.optim.SGD(net.parameters((),lr=0.03))

num_epochs=3
for epoch in range(num_epochs):

    for X,y in data_iter:
        l=loss(net(X),y)
        trainer.zero_grad()#梯度清零
        l.backward()
        trainer.step()#根据梯度更新参数
    l=loss(net(features),labels)
    print(f'epoch{epoch+1},loss{l:f}')