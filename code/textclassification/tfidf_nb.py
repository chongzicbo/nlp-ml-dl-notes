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
    """
    保存模型
    :param clf:
    :param savedPath:
    :return:
    """
    joblib.dump(clf, savedPath)


def loadModel(savedPath):
    """
    加载模型
    :param savedPath:
    :return:
    """
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
    print(model)
