---
title: linear SVM
date: 2023-09-26 01:02:42
tags:
---
# linear SVM
对SVM学习后，运行了线性内核（linear kernels）的向量机demo。

导入相关需要的包：
```py
import numpy as np
import pandas as pd
import sklearn.svm
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt
```
### 导入数据 load data
读取文件，导入数据。利用scipy.io中的loadmat函数读入mat文件，用keys()可以获得字典中所有的键,get()函数返回指定键的值,DataFrame()创建dataframe，参数columns为列名：
```py
mat = sio.loadmat('./data/ex6data1.mat')
print(mat.keys())

data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')#添加一列y
data.head()#打印出来前五行看看效果
```
![](./linear%20SVM/dataframe1.png)

```py
min(data['X1']), max(data['X1']), min(data['X2']), max(data['X2']) #获取data中x1的最大值和最小值，x2的最大值和最小值
```
## 可视化数据 visualize data

```py
positive = data[data.y == 1]#获取一个y列都为1的dataframe
negative = data[data.y == 0]#获取一个y列都为0的dataframe

fig, ax = plt.subplots(figsize=(8, 6)) #画图，siez为（width=8，height=6）
ax.scatter(positive['X1'], positive['X2'], label='positive', s=50, marker='+', c='r')#画子图，横纵坐标为positive的x1、x2，标签为positive，散点图中点大小为50，标记为'+',颜色为红色。
ax.scatter(negative['X1'], negative['X2'], label='negative', s=50, marker='o', c='b')#同上
ax.legend(loc='best')#图例位置为右上角
ax.set_xlabel('X1')
ax.set_ylabel('X2')#横纵坐标标签为x1,x2
plt.show()

```
## try c=1
选择模型sklearn库内的linearSVC模型，参数C为惩罚参数，loss表示损失函数，max_iter指定最大的迭代次数。
```py
svc1 = sklearn.svm.LinearSVC(C=1, loss='hinge', max_iter=20000)
svc1.fit(data[['X1', 'X2']], data['y'])#对数据进行训练
svc1.score(data[['X1', 'X2']], data['y'])#利用score()函数进行评估
```
### 决策边界 decision boundary
numpy库内的arange()函数用于生成数组，参数：(起始位置，终止位置，步长)。meshgrid(x,y) :基于向量x和y中包含的坐标返回二维网格坐标。
```py
fig, ax = plt.subplots(figsize=(8, 6))
positive = data[data.y == 1]
negative = data[data.y == 0]

ax.scatter(positive['X1'], positive['X2'], label='positive', s=50, marker='+', c='r')
ax.scatter(negative['X1'], negative['X2'], label='negative', s=50, marker='o', c='b')
ax.legend(loc='best')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# 决策边界, 使用等高线表示
x1 = np.arange(0, 4.5, 0.01)
x2 = np.arange(0, 5, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc1.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)]) #将x1、x2的值打包合并返回一个数组
plt.contour(x1, x2, y_pred, colors='g', linewidths=.5)#绘等高线

plt.show()
```
![](./linear%20SVM/output1.png)

### 直观显示样本到超平面的符号距离的不同。
```py
data['SVM1 Confidence'] = svc1.decision_function(data[['X1', 'X2']])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM1 Confidence'], cmap='seismic')#颜色随着距离的变化而不同
ax.set_title('SVM(C=1) Decision Confidence')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# 决策边界, 使用等高线表示
x1 = np.arange(0, 4.5, 0.01)
x2 = np.arange(0, 5, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc1.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)])
plt.contour(x1, x2, y_pred, colors='g', linewidths=.5)

plt.show()
```
![](./linear%20SVM/output2.png)

## try C=400
C越大越容易过拟合，图像中最左侧的点被划分到右侧。
```py
svc400 = sklearn.svm.LinearSVC(C=400, loss='hinge', max_iter=80000)
svc400.fit(data[['X1', 'X2']], data['y'])
svc400.score(data[['X1', 'X2']], data['y'])
```
### C=400 决策边界 decision boundary
```py
fig, ax = plt.subplots(figsize=(8, 6))
positive = data[data.y == 1]
negative = data[data.y == 0]

ax.scatter(positive['X1'], positive['X2'], label='positive', s=50, marker='+', c='r')
ax.scatter(negative['X1'], negative['X2'], label='negative', s=50, marker='o', c='b')
ax.legend(loc='best')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# 决策边界, 使用等高线表示
x1 = np.arange(0, 4.5, 0.01)
x2 = np.arange(0, 5, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc400.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)])
plt.contour(x1, x2, y_pred, colors='g', linewidths=.5)

plt.show()
```
![](./linear%20SVM/output3.png)

### C=400 直观显示样本到超平面的符号距离的不同。
```py
data['SVM400 Confidence'] = svc400.decision_function(data[['X1', 'X2']])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM400 Confidence'], cmap='seismic')
ax.set_title('SVM(C=400) Decision Confidence')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# 决策边界, 使用等高线表示
x1 = np.arange(0, 4.5, 0.01)
x2 = np.arange(0, 5, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc400.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)])
plt.contour(x1, x2, y_pred, colors='g', linewidths=.5)
```
![](./linear%20SVM/output4.png)
打印最终的dataframe查看数据，进行对比。
```py
data.head()
```
![](./linear%20SVM/dataframe2.png)
关于线性内核的SVM向量机demo运行学习就到此为止啦。