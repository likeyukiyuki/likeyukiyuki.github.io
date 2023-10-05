---
title: linear SVM
date: 2023-09-26 01:02:42
tags:
---
# linear SVM
支持向量机（support vector machines, SVM）是监督学习中一种二分类算法，它的原理是找到一个超平面划分两类样本。SVM不仅仅可以用于分类问题，也可以用于回归，但是通常还是用于二分类问题的。

/サポートベクターマシン(SVM)は、教師あり学習における二項分類アルゴリズムであり、2種類のサンプルを分割します超平面を見つけることによって機能します。 SVMは、分類問題だけでなく回帰にも使用できますが、二項分類問題によく使用されます。
![](./linear%20SVM/01.png)
## 核函数（kernel function）
SVM可以选择不同的核函数，也可以处理一些非线性的问题。常用的核函数有线性核函数、高斯核函数，一般来说，参数较少、并且线性可分时可以使用线性核函数，而参数比较多、线性不可分的时候可以使用高斯核函数。

/SVM は、異なるカーネル関数を選択したり、いくつかの非線形問題を処理したりできます。 一般的に使用されるカーネル関数は、線形カーネル関数、ガウスカーネル関数であり、一般に、線形カーネル関数は、パラメータが少なく線形に割り切れる場合に使用でき、ガウスカーネル関数は、より多くのパラメータと線形不可分性がある場合に使用できます。
## 软间隔　ソフトマージン
在我们划分样本时，很少能遇到完全线性的情况。
/サンプルを分割するとき、完全に線形な状況に遭遇することはめったにありません。
![](./linear%20SVM/02.png)
这时候我们就要放宽标准，允许一些在间隔带内的样本，也就是软间隔。
/このとき、標準を緩和し、スペーサーバンド内のいくつかのサンプルがあることを許可し、つまり、ソフトマージンです。
![](./linear%20SVM/03.png)
而对于错误样本的容忍程度在线性SVM模型中可以用参数C来设置，C是一个惩罚参数，当C越大时，样本对错误参数的容忍程度越低，C接近无穷大时，会变完全回线性的SVM；而当C为较小的有限值时，才会允许一些误差值存在。
/そして、線形SVMモデルの誤差サンプルの許容度は、パラメータCで設定することができます、Cはペナルティパラメータであり、Cが大きくなると、サンプルは誤差パラメータの許容度が小さくなります、Cが無限大に近づくと、線形SVMに完全に戻ってしまいます；Cがより小さい有限の値になると、その時だけ、いくつかの誤差値が存在することを許容します。
## SVM linearSVC demo
对SVM学习后，运行了线性内核（linear kernels）的向量机demo。

/SVMについて学んだ後、線形カーネル（linear kernels）を使ったＳＶＭのデモが実行された。

导入相关需要的包/関連する必要なパッケージをインポートします：
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

/ファイルを読み込み、データをインポートします。 loadmat 関数で scipy.io を使用して mat ファイルを読み込み、keys () で辞書内のすべてのキーを取得し、get () 関数で指定したキーの値を返します、DataFrame() はデータフレームを作成します、パラメータ columns にはカラム名を指定します：：
```py
mat = sio.loadmat('./data/ex6data1.mat')
print(mat.keys())

data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
data['y'] = mat.get('y')#添加一列y /y列を追加
data.head()#打印出来前五行看看效果 /最初の5行をプリントアウトして効果を見ます
```
![](./linear%20SVM/dataframe1.png)

```py
min(data['X1']), max(data['X1']), min(data['X2']), max(data['X2']) #获取data中x1的最大值和最小值，x2的最大值和最小值 /データ中のx1の最大値と最小値、およびデータ中のx2の最大値と最小値を取得します。
```
## 可视化数据 visualize data

```py
positive = data[data.y == 1]#获取一个y列都为1的dataframe /y列がすべて1のデータフレームを取得します。
negative = data[data.y == 0]#获取一个y列都为0的dataframe /y列がすべて0のデータフレームを取得します。

fig, ax = plt.subplots(figsize=(8, 6)) #画图，siez为（width=8，height=6） /ドローイング、シーズは（width=8, height=6）
ax.scatter(positive['X1'], positive['X2'], label='positive', s=50, marker='+', c='r')#画子图，横纵坐标为positive的x1、x2，标签为positive，散点图中点大小为50，标记为'+',颜色为红色。 /x1,x2を正の水平座標と垂直座標とし、ラベルはpositive、点サイズ50の散布図、'+'のラベル、赤色で描画します。
ax.scatter(negative['X1'], negative['X2'], label='negative', s=50, marker='o', c='b')#同上 /同上
ax.legend(loc='best')#图例的位置为右上角 /凡例は右上隅に配置されます
ax.set_xlabel('X1')
ax.set_ylabel('X2')#横纵坐标标签为x1,x2 /x1,x2とラベル付けされた水平および垂直座標
plt.show()

```
## try c=1
因为我们需要选择一个线性分类模型，所以选择模型sklearn库内的linearSVC模型。linear代表线性，SVC代表分类模型。sklearn库内的svm模型还有很多，例如SVR是回归模型。
参数C为惩罚参数，loss表示损失函数，max_iter指定最大的迭代次数。

/線形分類モデルを選択する必要があるため、モデル sklearn ライブラリ内でlinearSVC モデルを選択します。 線形は線形を表し、SVCは分類モデルを表します。 sklearnライブラリにはさらに多くのSVMモデルがあり、たとえば、SVRは回帰モデルです。
パラメーター C はペナルティ パラメーター 、lossは損失関数を表し、max_iterは最大反復回数を指定します。
```py
svc1 = sklearn.svm.LinearSVC(C=1, loss='hinge', max_iter=20000)
svc1.fit(data[['X1', 'X2']], data['y'])#对数据进行训练 /データのトレーニング
svc1.score(data[['X1', 'X2']], data['y'])#利用score()函数进行评估 /score()関数による評価
```
### 决策边界 decision boundary
numpy库内的arange()函数用于生成数组，参数：(起始位置，终止位置，步长)。meshgrid(x,y) :基于向量x和y中包含的坐标返回二维网格坐标。

/numpy ライブラリで、配列を生成するための range () 関数、パラメータ: (開始位置、終了位置、ステップサイズ)。meshgrid(x,y): ベクトル x と y に含まれる座標に基づいて 2 次元グリッド座標を返します。
```py
fig, ax = plt.subplots(figsize=(8, 6))
positive = data[data.y == 1]
negative = data[data.y == 0]

ax.scatter(positive['X1'], positive['X2'], label='positive', s=50, marker='+', c='r')
ax.scatter(negative['X1'], negative['X2'], label='negative', s=50, marker='o', c='b')
ax.legend(loc='best')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# 决策边界, 使用等高线表示 /等高線で表された決定境界線
x1 = np.arange(0, 4.5, 0.01)
x2 = np.arange(0, 5, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc1.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)]) #将x1、x2的值打包合并返回一个数组 进行预测（predict）/x1, x2 の値を詰め合わせて配列を返す、予測を行う。
plt.contour(x1, x2, y_pred, colors='g', linewidths=.5)#绘等高线 /等高線を描く

plt.show()
```
![](./linear%20SVM/output1.png)

### 直观显示样本到超平面的符号距离的不同。 /サンプルから超平面までのシンボル距離の差を視覚化します。
```py
data['SVM1 Confidence'] = svc1.decision_function(data[['X1', 'X2']])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM1 Confidence'], cmap='seismic')#颜色随着距离的变化而不同 /色は距離によって異なります
ax.set_title('SVM(C=1) Decision Confidence')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# 决策边界, 使用等高线表示 /等高線で表された決定境界線です
x1 = np.arange(0, 4.5, 0.01)
x2 = np.arange(0, 5, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc1.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)])
plt.contour(x1, x2, y_pred, colors='g', linewidths=.5)

plt.show()
```
![](./linear%20SVM/output2.png)

## try C=400
前面说过，C越大时对间隔带内存在样本的容忍度越低，所以C越大越容易过拟合，图像中最左侧的点被划分到右侧。 /前述のように、Cが大きいほどスペーサーバンド内のサンプルの許容誤差が低くなるため、Cが大きいほどオーバーフィットしやすくなり、画像の左端のポイントが右に分割されます。
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

# 决策边界, 使用等高线表示 /等高線で表された決定境界線
x1 = np.arange(0, 4.5, 0.01)
x2 = np.arange(0, 5, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc400.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)])
plt.contour(x1, x2, y_pred, colors='g', linewidths=.5)

plt.show()
```
![](./linear%20SVM/output3.png)

### C=400 直观显示样本到超平面的符号距离的不同。 /サンプルから超平面までのシンボル距離の差を視覚化すます。
```py
data['SVM400 Confidence'] = svc400.decision_function(data[['X1', 'X2']])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM400 Confidence'], cmap='seismic')
ax.set_title('SVM(C=400) Decision Confidence')
ax.set_xlabel('X1')
ax.set_ylabel('X2')

# 决策边界, 使用等高线表示  /等高線で表された決定境界線
x1 = np.arange(0, 4.5, 0.01)
x2 = np.arange(0, 5, 0.01)
x1, x2 = np.meshgrid(x1, x2)
y_pred = np.array([svc400.predict(np.vstack((a, b)).T) for (a, b) in zip(x1, x2)])
plt.contour(x1, x2, y_pred, colors='g', linewidths=.5)
```
![](./linear%20SVM/output4.png)
打印最终的dataframe查看数据，进行对比。 /最終dataframeを印刷して、比較のためのデータを表示すます。
```py
data.head()
```
![](./linear%20SVM/dataframe2.png)
关于线性内核的SVM向量机demo运行学习就到此为止啦。/線形カーネルで動作しますSVMベクトルマシンのデモの研究はこれで終わりです。