import random
import torch
from d2l import torch as d2l
#带有噪声的线性模型 y=Xw+b+噪声值
def synthetic_data(w,b,num_examples):
    X=torch.normal(0,1,(num_examples,len(w)))#生成服从标准正态分布的随机张量，元素均值为0，标准差为1，张量的形状是num_examples*len(w)
    Y=torch.matmul(X,w)+b#matmul张量乘法
    Y+=torch.normal(0,0.01,Y.shape)
    return X,Y.reshape((-1,1))#线性回归强制输出值Y为列向量，二维

true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)
print('features:',features[0],'\nlabel:',labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
d2l.plt.show()#d2l.plt 是 matplotlib.pyplot 的一个别名，而 matplotlib 默认在非交互式环境中（比如 PyCharm 或脚本）绘制图形时并不会自动显示
#没有 d2l.plt.show()，你绘制的图形虽然已经存在于内存中，但不会在屏幕上显示出来。

#小批量（mini-batch）的方式随机读取数据
def data_iter(batch_size,features,labels):
    num_examples=len(features)#获取样本总数
    indices=list(range(num_examples))#创建一个从0-总数-1的列表，代表每个样本的索引
    #对索引列表进行随机打乱，保证每次迭代读取的数据是随机的
    random.shuffle(indices)

    #遍历打乱后的索引列表
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)])#要是加上步长大于长度了就选择总长度
        yield features[batch_indices],labels[batch_indices] #yield是生成器，每次循环都返回一个元组

batch_size=10

for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break

#初始化模型参数
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)
def linreg(X,w,b):
    """线性回归模型"""
    return torch.matmul(X,w)+b #用来计算预测值的一个公式

#定义损失函数
def squared_loss(y_hat,y):
    """均方损失"""
    #y_hat是预测值，y是真实价格，你要把y的形状调了才能做差，均方损失是预测值减去真实值平方
    return (y_hat-y.reshape(y_hat.shape))**2/2

#定义优化算法，模型学习和自我纠错的核心技术
def sgd(params,lr,batch_size):#lr是学习率，控制了每次更新参数的步长，学习率太大会导致模型跳过最优解，太小练得慢
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:#param是需要更新的参数，之前定义的w和b
            param-=lr*param.grad/batch_size #随机梯度下降的核心更新公式，沿着梯度的反方向以学习率为步长更新参数
            #-号代表了下山的意思，到山底就是损失函数最小
            #batch_size为何要取平均？？
            #！！！其实梯度只是代表了方向，10 5 2是从不同维度指明它的方向，而不是连下山的多少大小都说明，我们对/是为了让下山的方向更准确，而控制下山的步伐的是学习率
            param.grad.zero_()#清除梯度

lr=0.03
num_epochs=3
net=linreg
loss=squared_loss

for epoch in range(num_epochs):
    #遍历小批量
    for X,y in data_iter(batch_size,features,labels):
        l=loss(net(X,w,b),y) #计算预测值
        l.sum().backward() #计算梯度，损失对w的梯度存在了对w.grad中，损失对b的梯度存储在b.grad中
        sgd([w,b],lr,batch_size) #更新参数，传入梯度(纠正的方向)、学习率等
    with torch.no_grad(): #计算打印模型的损失
        train_l=loss(net(features,w,b),labels)
        print(f'epoch{epoch+1},loss{float(train_l.mean()):f}')