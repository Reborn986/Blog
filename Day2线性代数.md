#### Tensor（张量）的本质

张量tensor是任意维度的数组，标量、向量、矩阵都是其中的特殊情况

标量：单个数（如a）  向量：一列数，[a,b,c]   矩阵：二维表格（[[a,b],[c,d]]）

##### 用tensor表示标量

```python
x=torch.tensor([3.0])
```



##### 用tensor表示向量

向量就是用标量值组成的列表

```python
x=torch.arange(4)#arange本身就是生成一个一维的列表

```

###### 通过张量的索引来访问向量中的任意一个元素

```python
x[3]
```

###### 张量的长度

```python
len(x) #4
```

###### 张量的形状

只有一个轴的向量，形状只有一个元素

```python
x.shape #torch.Size([4])
####两个轴的话...
m=torch.tensor([[1,2],[5,6]])
print(m.shape)
#torch.Size([2, 2])
```



##### 用tensor表示矩阵

###### 利用arange和reshape生成

arange是生成一个一维列表，只有所有的元素，reshape改变其中的形状

```python
A=torch.arange(20).reshape(5,4)#reshape改成五行四列的
print(A)
```

###### 转置

```python
print(A.T)#不带括号
```

高维张量没有转置

######  对称矩阵

```python
B==B.T#print出来是一个tensor,每一个元素是True或者False
```

##### 三维及以上的张量

##### 三维张量

```python
A=torch.arange(24).reshape(2,3,4)
#tensor([[[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11]],

#        [[12, 13, 14, 15],
#         [16, 17, 18, 19],
#         [20, 21, 22, 23]]])
```

3 4表示三行四列，2表示两个，维度体现在[]

书架有2层，每一层有3行，每一列有4列



#### 张量的操作

###### 分配新内存分配副本

```python
B=A.clone()
```

###### 哈达玛积

两个矩阵按照元素乘法

```python
A*B
```

###### 指定求和

```python
A=torch.arange(24).reshape(2,3,4)
print(A)
#tensor([[[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]],

#        [[12, 13, 14, 15],
#         [16, 17, 18, 19],
#         [20, 21, 22, 23]]])
```

 把第一个维度求和,用axis=指定

```python
A_sum_axis0=A.sum(axis=0)
print(A_sum_axis0)
#tensor([[12, 14, 16, 18],
#        [20, 22, 24, 26],
#        [28, 30, 32, 34]])
print(A_sum_axis0.shape)
#torch.Size([3, 4])
```

 把第二个维度求和

2保留着，但是两个的三行都相加了

```python
#tensor([[12, 15, 18, 21],
#        [48, 51, 54, 57]])
#如果想保留原来的维度并且另原来的维度是（还是3个[]但是里面不包含别的）
A_sum_axis1 = A.sum(axis=1,keepdims=True)#这样sum后和原来的维度还是一样的
print(A_sum_axis1)
#tensor([[[12, 15, 18, 21]],

#        [[48, 51, 54, 57]]])
```

 一二维度求和和二三维度不一样

只有一二求和才等于直接sum,说明sum一直是行之间的加

计算求和的时候那个被求和的维度的大小为1（广播机制的时候会用）

```python

```



###### 均值

```python
A.mean()
```

###### 点积

dot

###### 矩阵相乘

5x4的和4x1的

```python
torch.mm(A,B)
```

###### l2范数

范数是向量元素平方和的平方根

```python
torch.norm(A)
```

###### l1范数

向量元素的绝对值之和

```python
torch.abs(A).sum()
```

###### 矩阵的F范数

矩阵的元素的平方求和再开根

#### 矩阵的计算（对矩阵求导数）

![image-20250805184202192](C:/Users/10842/AppData/Roaming/Typora/typora-user-images/image-20250805184202192.png)



x和y都是矩阵

![image-20250805184249515](C:/Users/10842/AppData/Roaming/Typora/typora-user-images/image-20250805184249515.png)

求导是按列优先进行求导，然后按行优先放到结果矩阵里