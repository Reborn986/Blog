标量不变，y横向拉伸，x纵向拉伸

###### 求导公式积累

1.x转置*x=torch.dot（x,x）

左半边×的是一个数，右面你想（x1,x2,x3）*(y1,y2,y3)点积不就是x1y1+x2y2+x3y3吗

2.

#### 自动求导原理

计算图+反向传播

反向传播就是链式法则

#### 代码实现

1.建立一个储存张量并且开始对自变量进行梯度追踪

```python
import torch
x=torch.arange(4.0)
print(x)
x.requires_grad_(True) #开启对 x 的梯度追踪，反向传播的前提，x是自变量
print(x.grad)#打印出来是none
y=2*torch.dot(x,x)
```

2.反向传播

```python
y.backward()
#[4x0,4x1,4x2,4x3]
```

3.进行新的计算时必须清零，pytorch会自动储存

```python
x.grad.zero_()
```

4.进行新的追踪，sum为例子

```python
y=x.sum()#y=x0+x1+x2...
y.backward()#对x0求导第一个是1，对x求导也是1，所以总的是1 1 1 1
print(x.grad)
```

5.detach

剪短某一个通路

```
x.grad.zero_()
y=x*x
u=y.detach()
z=u*x
print(z.sum().backward())
print(x.grad==u)
```

假如说没有detach,z.sum是 x1^3+x2^3+x3^3,然后backward一次是[3x1^2,3x2^2,3x3^2]

<img src="../../../../xwechat_files/wxid_cws7a0ywqqzw22_519f/temp/RWTemp/2025-08/029c93af093db99272706ad2d5826771.jpg" alt="029c93af093db99272706ad2d5826771" style="zoom: 25%;" />