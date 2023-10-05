---
title: KNN算法（K Nearest Neighbors）/k近傍法
date: 2023-09-28 01:32:02
tags:
---
## KNN算法（K Nearest Neighbors） 理论介绍
之前进行了k-means算法的学习，而接下来介绍的KNN算法（K Nearest Neighbors）与其有些相似，虽然都可以进行分类，但是KNN是监督学习，理论也较为简单。K Nearest Neighbors直译为K个最近的邻居，而KNN的工作原理就是输入一个特征向量x后，只选择样本数据集中与x最相似的k个数据，然后把x的类别预测为这k个样本中类别数最多的那一类。KNN算法最简单粗暴的就是将预测点与所有点距离进行计算，然后保存并排序，选出前面 K 个值看看哪些类别比较多。KNN也可以用于回归预测，同样是寻找距离最近的k个样本，然后对这k个样本的目标值去均值即可作为新样本的预测值。

/KNNアルゴリズム（K Nearest Neighbors）は、以前に紹介したk-meansアルゴリズムと多少似ているが、どちらも分類を行うことができるが、KNNは教師あり学習であり、理論はより単純である。k Nearest NeighborsはK最近傍と訳され、KNNの原理は、特徴ベクトルxを入力した後サンプルデータセットの中からxに最も似ているk個のデータだけを選び、そのk個のサンプルの中で最もカテゴリー数が多いものをxのカテゴリーと予測する。KNNアルゴリズムの最も単純で粗雑な部分は、予測された点とすべての点の間の距離を計算し、どのカテゴリーが多いかを見るために最初のK個の値を保存して並べ替えることである。 KNNは回帰予測にも使用でき、同じように最も近いk個のサンプルを見つけ、その平均に対するk個のサンプルの目標値を新しいサンプルの予測値として使用できる。
## KNN demo
接下来通过一个分类预测的demo来解释KNN的基本原理及过程 /次に、分類予測のデモを通して、KNNの基本原理とプロセスを説明する：
```py
#导入所需要的包 /必要なパッケージをインポートする
from sklearn import datasets
from collections import Counter 
from sklearn.model_selection import train_test_split
import numpy as np
```
导入数据和标签，划分训练集和测试集，并且利用train_test_split()函数打乱，其中参数未划分的数据集X，未划分的标签y，随机数种子random_state=2003，应用于分割前对数据的洗牌。

/データとラベルがインポートされ、トレーニングセットとテストセットが分割され、train_test_split()関数を用いて、セグメンテーションされていないデータセットX、セグメンテーションされていないラベルy、乱数シード（random_state=2003）は、セグメンテーションの前にデータをシャッフルするために使用する。
```py
# 导入iris数据 / irisデータのインポート
iris = datasets.load_iris() 
X = iris.data #数据集 /データセット
y = iris.target #标签 /タブ
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2003)
```
计算两个样本之间的距离 /2つのサンプル間の距離を計算する
```py
def euc_dis(instance1, instance2):
    
    dist = np.sqrt(np.sum((instance1-instance2)**2)) #对instance1和instance2求差的平方和，即利用欧式公式求距离
    return dist
```
```py
def knn_classify(X, y, testInstance, k):
    """
    给定一个测试数据testInstance, 通过KNN算法来预测它的标签。 /テストデータtestInstanceが与えられると、そのラベルはKNNアルゴリズムによって予測される。
    X: 训练数据的特征 /トレーニングデータの特徴
    y: 训练数据的标签 /トレーニングデータのラベル
    testInstance: 测试数据，这里假定一个测试数据 array型 / テストデータ、ここではテストデータを想定 array
    k: 选择多少个neighbors? /いくつのneighborsから選べますか？
    """

    # 计算 testInstance 与 X的距离 /testInstanceとXの距離を計算する。
    dists=[euc_dis(x,testInstance) for x in X]
   
    # 找出最近的K个元素的idx /最も近いK個の要素のidxを求める
    idxknn= np.argsort(dists)[:k] #将dists从小到大排序，返回排序后的元素 /distsを小さいものから大きいものへとソートし、ソートされた要素を返す

    # 找出KNN对应的n个y值 /KNNに対応するn個のy値を求める
    yknn=y[idxknn]

    # 返回数组中出现最多次数的值 /配列中の出現回数が最も多い値を返します。
    return Counter(yknn).most_common(1)[0][0]
```
预测结果 /予想される結果：
```py 
predictions = [knn_classify(X_train, y_train, data, 3) for data in X_test] #遍历测试集中数据，并且通过KNN得到其对应标签
correct = np.count_nonzero((predictions==y_test)==True) #将预测标签与测试集标签进行对比，得到正确的标签个数 /正しいラベル数を得るために、予測されたラベルとテストセットのラベルを比較する
print ("Accuracy is: %.3f" %(correct/len(X_test))) #通过len()得到测试集标签个数，相除得到准确率 /使用 len() 获取测试集中的标签数量，然后除以正确率
```
KNN算法的demo讲解就到此为止啦~ / これでKNNアルゴリズムのデモは終わりである