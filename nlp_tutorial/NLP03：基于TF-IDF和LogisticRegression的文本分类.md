# 1.TF-IDF算法步骤

## 1.1 计算词频

![](https://pic3.zhimg.com/v2-281a550de928afe343c055d06371cf77_r.jpg)

考虑到文章有长短之分，为了便于不同文章的比较，进行"词频"标准化。

![](https://pic1.zhimg.com/v2-393435b342546a2f1736d1d755adb1cd_r.jpg)

## 1.2 计算逆文档频率

需要一个语料库（corpus），用来模拟语言的使用环境。

![](https://pic2.zhimg.com/v2-1d5c436e04f497544d72fec6909a3fad_r.jpg)

如果一个词越常见，那么分母就越大，逆文档频率就越小越接近0。分母之所以要加1，是为了避免分母为0（即所有文档都不包含该词）。log表示对得到的值取对数。

## 1.3 计算TF-IDF

![](https://pic2.zhimg.com/v2-5560a4b2efa3330021b8b2ef13a471fe_r.jpg)

可以看到，TF-IDF与一个词在文档中的出现次数成正比，与该词在整个语言中的出现次数成反比。所以，自动提取关键词的算法就很清楚了，就是**计算出文档的每个词的TF-IDF值，然后按降序排列，取排在最前面的几个词。**

## 1.4 **优缺点**

TF-IDF的优点是简单快速，而且容易理解。缺点是有时候用**词频**来衡量文章中的一个词的重要性不够全面，有时候重要的词出现的可能不够多，而且这种计算无法体现位置信息，无法体现词在上下文的重要性。如果要体现词的上下文结构，那么你可能需要使用word2vec算法来支持。

# 2. LogisticRegression基本原理

## 2.1 什么是LR

logistic回归虽然说是回归，但确是为了解决分类问题，是二分类任务的首选方法，简单来说，输出结果不是0就是1。逻辑回归（Logistic Regression）与线性回归（Linear Regression）都是一种广义线性模型（generalized linear model）。

逻辑回归假设因变量 y 服从二项分布，而线性回归假设因变量 y 服从高斯分布。

因此与线性回归有很多相同之处，去除Sigmoid映射函数的话，逻辑回归算法就是一个线性回归。

可以说，逻辑回归是以线性回归为理论支持的，但是逻辑回归通过Sigmoid函数引入了非线性因素，因此可以轻松处理0/1分类问题。

换种说法：

线性回归，直接可以分为两类，

但是对于图二来说，在角落加上一块蓝色点之后，线性回归的线会向下倾斜，参考紫色的线，

但是logistic回归（参考绿色的线）分类的还是很准确，logistic回归在解决分类问题上还是不错的

![img](https://img2018.cnblogs.com/i-beta/1750260/201911/1750260-20191128081134667-2120619535.png)![img](https://img2018.cnblogs.com/i-beta/1750260/201911/1750260-20191128081353058-1054584462.png)

## 2.2 LR原理

Sigmoid函数：

曲线：![img](https://img2018.cnblogs.com/common/1750260/201911/1750260-20191127211750890-2025272555.png)

![img](https://img2018.cnblogs.com/common/1750260/201911/1750260-20191127211647287-584640389.png)

 

之后推导公式中会用到：

 ![img](https://img2018.cnblogs.com/i-beta/1750260/201911/1750260-20191128091940142-302514267.png)

我们希望随机数据点被正确分类的概率最大化，这就是最大似然估计。

最大似然估计是统计模型中估计参数的通用方法。

你可以使用不同的方法（如优化算法）来最大化概率。

牛顿法也是其中一种，可用于查找许多不同函数的最大值（或最小值），包括似然函数。也可以用梯度下降法代替牛顿法。

既然是为了解决二分类问题，其实也就是概率的问题，分类其实都是概率问题。

假定：

y=1和y=0的时候的概率

![img](https://img2018.cnblogs.com/i-beta/1750260/201911/1750260-20191128081902330-1197501799.png) ![img](https://img2018.cnblogs.com/i-beta/1750260/201911/1750260-20191128082030762-325715555.png)

 

 似然函数：其实就是概率相乘，然后左右两边同时取对数

![img](https://img2018.cnblogs.com/i-beta/1750260/201911/1750260-20191128082325232-402902342.png)

![img](https://img2018.cnblogs.com/i-beta/1750260/201911/1750260-20191128082459322-1863885947.png)

 

 

对数似然函数，求导，得到θ的梯度

![img](https://img2018.cnblogs.com/i-beta/1750260/201911/1750260-20191128082835128-2030177374.png)

 

 因为P=g(θX),P其实是θ的函数，X已知，要想P越大，就要θ越大，梯度上升问题

得到θ的学习规则：α为学习率

![img](https://img2018.cnblogs.com/i-beta/1750260/201911/1750260-20191128083504062-1083231453.png)

 

 最后将θ带入h(x)函数，求出概率

![img](https://img2018.cnblogs.com/i-beta/1750260/201911/1750260-20191128083718313-707906965.png)

 

总结来说：

比较一下logistic回归的参数学习规则和线性回归的参数学习规则

两个都是如下，形式一样，只是不同的是

线性回归  h(x)=θX

logistic回归 ![img](https://img2018.cnblogs.com/i-beta/1750260/201911/1750260-20191128083718313-707906965.png)

一个使用的模型是线性函数，一个使用的是sigmoid函数

![img](https://img2018.cnblogs.com/i-beta/1750260/201911/1750260-20191128083504062-1083231453.png)

# 3.使用TF-IDF和LR进行文本分类

```python
import os
import jieba
import re
import pickle
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

"""
基于tfidf和朴素贝叶斯或者LogisticRegression的文本分类
"""


def textSegment(filePath):
    """
    读取文本文件并进行分词
    :param filePath:文件路径
    :return:
    """
    textLines = open(filePath, "r", encoding="utf-8").read()
    textLines = re.sub("[^\u4e00-\u9fa5]", "", textLines)  # 只保留中文
    textLines = jieba.cut(textLines)
    return " ".join(textLines)


def loadTextFiles(fileDir, label):
    """
    获取指定目录下的文件及相应的标签,每个文件表示一个训练或测试样本，并进行相应地分词
    :param fileDir:文件目录
    :param label:标签
    :return:
    """
    textFiles = os.listdir(fileDir)
    textLineList = []
    labels = []

    for file in textFiles:
        filePath = fileDir + "/" + file
        textLineList.append(textSegment(filePath))
        labels.append(label)
    return textLineList, labels


def loadTrainDataset(fileDir, countVectorizer, use_bow=False):
    """
    :param fileDir: 训练文件目录
    :param countVectorizer: CountVectorizer类
    :param use_bow:默认为False，使用tfidf特征作为输入，否则使用词袋向量
    :return:
    """
    textLineList1, labels1 = loadTextFiles(fileDir + "hotel", "宾馆")
    textLineList2, labels2 = loadTextFiles(fileDir + "travel", "旅游")
    textLineList = textLineList1 + textLineList2
    labels = labels1 + labels2

    vectorMatrix = countVectorizer.fit_transform(textLineList)
    if not use_bow:
        vectorMatrix = TfidfTransformer(use_idf=False).fit_transform(vectorMatrix)
    return vectorMatrix, labels


def train(vectorMatrix, labels, modelType="NB"):
    """
    模型训练
    :param vectorMatrix:特征
    :param labels:标签
    :param modelType:模型类型，默认为朴素贝叶斯
    :return:
    """
    # LogisticRegression
    if modelType == "LR":
        clf = LogisticRegression(penalty="l2")
    else:
        clf = MultinomialNB()
    clf.fit(vectorMatrix, labels)
    return clf


def saveModel(clf, savedPath):
    joblib.dump(clf, savedPath)


def loadModel(savedPath):
    return joblib.load(savedPath)


if __name__ == '__main__':
    # filePath = "F:\\data\\machine_learning\\分类数据\\dataset\\test\\hotel\\xm7_seg_pos.txt"
    # textLines = textSegment(filePath)
    # print(textLines)

    fileDir = "F:\\data\\machine_learning\\分类数据\\dataset\\train\\"
    # textLineList1, labels1 = loadTextFiles(fileDir + "hotel", "宾馆")
    # textLineList2, labels2 = loadTextFiles(fileDir + "travel", "旅游")
    # print(len(textLineList1), len(textLineList2), len(labels1), len(labels2))
    # print(textLineList1)

    # countVectorizer = CountVectorizer()
    # vectorMatrix, labels = loadTrainDataset(fileDir, countVectorizer)
    # clf = train(vectorMatrix, labels, modelType="LR")
    # saveModel(clf, savedPath="../../../models/lr_model.m")

    model = loadModel(savedPath="../../../models/lr_model.m")
```

代码：https://github.com/chongzicbo/nlp-ml-dl-notes/blob/master/code/textclassification/tfidf_nb.py

# 参考：

[1]https://zhuanlan.zhihu.com/p/31197209

[2]https://www.cnblogs.com/xiuercui/p/11945567.html