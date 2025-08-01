处理的NaN的目的是如果有NaN没法转成张量

### 数字类插平均值取代NaN

#### 插值报错

###### 报错显示

按照李沐老师的代码手敲一遍后，运行报错

```python
import os
os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file=os.path.join('..','data','house_tiny.csv')

#手工创建一个表格
with open(data_file,'w') as f:#代码结束后关闭这个模块
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd

data=pd.read_csv(data_file)
print(data)

#inputs、outputs分别输入数据
#inputs输入所有行不包括第二列，outputs是第二列price
inputs,outputs=data.iloc[:,0:2],data.iloc[:,2]
inputs=inputs.fillna(inputs.mean())
print(inputs)
```

报错如下，显示不能把int和str连接，只能把str和str连接

```python
Traceback (most recent call last):
  File "G:\numpy\data_process.py", line 20, in <module>
    inputs=inputs.fillna(inputs.mean())
TypeError: can only concatenate str (not "int") to str
```

###### 报错分析

报错是因为pandas库的调整导致的：

.mean()是按照列来处理，第一列被识别为数字类型，第一列求出3后可以填充NA

第二列由于PAVE被识别为str类型，数字3不能填充str

###### 解决方案

在mean()添加：numeric_only=True

这样只计算数值类的平均值，这样只有第一列数值类型会被填充

#### 结果分析

![image-20250801171445038](Day1%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.assets/image-20250801171445038.png)

```python
nputs=inputs.fillna(inputs.mean(numeric_only=True))
```

该行代码让第一列的NaN填充了数字平均值，第二列无变化

### 分类标记法处理字符串

还有第一列的PAVE和NaN没有处理，使用get_dummies()将Pave和Nan分类

```python
#NaN记为特征0
inputs=pd.get_dummies(inputs,dummy_na=True).astype(int) #否则显示的是false和true，而不是0或者1
print(inputs)

```

结果如下

![image-20250801172250212](Day1%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.assets/image-20250801172250212.png)

### 重点函数

#### os库

operating system,提供和操作系统交互的函数，比如文件/目录操作、进程管理、环境变量管理

##### os.makedirs(路径名称，存在与否 exist_ok),返回none

exist_ok=True是一个可选参数,默认值为 False.exist_ok=True：如果要创建的目录已经存在，不会报错，程序会继续正常执行。

exist_ok=False（默认）：如果要创建的目录已经存在，会抛出 `FileExistsError` 异常。

假设你的当前工作目录是：/home/user/project/

os.path.join('..', 'data')会生成路径：/home/user/project/../data

这里的 `..` 指向 `/home/user/`，所以最终路径等价于：/home/user/data

`..` 是一个特殊符号，表示**上级目录**（即当前目录的父目录）

##### os.path.join()

`os.path.join()` 接受**任意数量的字符串参数**，将它们拼接成一个完整的路径.它会根据操作系统的不同自动选择正确的路径分隔符（`/` 或 `\`），并处理多余的斜杠等问题

#### get_dummies()

将分类数据转换为独热编码（0或者1），每一个特征新增一列，有为1无为0

```python
pandas.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)
```

data是必须参数，对什么进行分类就输入

**`dummy_na`**: 如果设置为 `True`，会将 `NaN` 值也作为一个独立的类别进行编码

#### fillna和mean

```python
inputs=inputs.fillna(inputs.mean(numeric_only=True))
```

`fillna()` 的主要作用是**用指定的值或方法来填充数据中的缺失值（NaN）**。

`mean()` 的主要作用是**计算数据中所有值的平均值**

**`numeric_only`**：**这是你在代码中使用的关键参数**。如果设置为 `True`，`mean()` 只会计算数字类型列的平均值，并忽略非数字列（如字符串列）

### 练习

创建包含更多行和列的原始数据集。

1. 删除缺失值最多的列。
2. 将预处理后的数据集转换为张量格式。

##### 创建原始数据集

```python
import os
os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file2=os.path.join('..','data','house_tiny.csv')
with open(data_file2,'w') as f:#代码结束后关闭这个模块
    f.write('NumRooms,Alley,Tom,Price\n')
    f.write('NA,Pave,NA,127500\n')
    f.write('2,NA,NA,106000\n')
    f.write('4,NA,NA,178100\n')
    f.write('NA,NA,Pave,140000\n')
    f.write('6,Pave,NA,140000\n')
```

##### 利用isnull区分NaN、统计最多缺失值

```python
missing_counts = data2.isnull().sum()#直接把每一列的NaN统计出来了
most_missing = missing_counts.idxmax()#比较最多的
inputs_cleaned = data2.drop(columns=most_missing)#drop方法删除列
print("删除缺失值最多的列后的数据:")
print(inputs_cleaned)
```

##### 把NaN代替或者分类处理

如果还有NaN是无法转化成张量的

```python
inputs_cleaned=inputs_cleaned.fillna(inputs_cleaned.mean(numeric_only=True))
inputs_cleaned=pd.get_dummies(inputs_cleaned,dummy_na=True).astype(int)

```

##### 转化为张量

```python
import torch
X,y=torch.tensor(inputs_cleaned.values),torch.tensor(data2.iloc[:,3].values) #前两列和最后一列的Price分别处理
print(X,y)

```

### 源码

###### 讲课源码（debug后）

```python
import os
os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file=os.path.join('..','data','house_tiny.csv')

#手工创建一个表格
with open(data_file,'w') as f:#代码结束后关闭这个模块
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd

data=pd.read_csv(data_file)
print(data)

#数字类型的NaN填充平均值
inputs,outputs=data.iloc[:,0:2],data.iloc[:,2]
inputs=inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

#NaN记为特征0
inputs=pd.get_dummies(inputs,dummy_na=True).astype(int) #否则显示的是false和true，而不是0或者1
print(inputs)

#将CSV文件转为张量
import torch
X,y=torch.tensor(inputs.values),torch.tensor(outputs.values) #前两列和最后一列的Price分别处理
print(X,y)
```

###### 作业

```python
############ 作业 #################################
import os
os.makedirs(os.path.join('..','data'),exist_ok=True)
data_file2=os.path.join('..','data','house_tiny.csv')
with open(data_file2,'w') as f:#代码结束后关闭这个模块
    f.write('NumRooms,Alley,Tom,Price\n')
    f.write('NA,Pave,NA,127500\n')
    f.write('2,NA,NA,106000\n')
    f.write('4,NA,NA,178100\n')
    f.write('NA,NA,Pave,140000\n')
    f.write('6,Pave,NA,140000\n')

import pandas as pd
data2=pd.read_csv(data_file2)
print(data2)

#处理缺失值最多的列：1.第一列是数字 第二三列是object，如果进行分类编码应该分开处理，分开编码，或者NaN一类其它的一类？
missing_counts = data2.isnull().sum()
most_missing = missing_counts.idxmax()
inputs_cleaned = data2.drop(columns=most_missing)
print("删除缺失值最多的列后的数据:")
print(inputs_cleaned)

#转化为张量，直接这样转化是错的，张量里面不能有NaN
#import torch
#X,y=torch.tensor(inputs_cleaned.values),torch.tensor(data2.iloc[:,3].values) #前两列和最后一列的Price分别处理
# print(X,y)

#先处理缺失值
inputs_cleaned=inputs_cleaned.fillna(inputs_cleaned.mean(numeric_only=True))
inputs_cleaned=pd.get_dummies(inputs_cleaned,dummy_na=True).astype(int)

#转化为张量
import torch
X,y=torch.tensor(inputs_cleaned.values),torch.tensor(data2.iloc[:,3].values) #前两列和最后一列的Price分别处理
print(X,y)

```

