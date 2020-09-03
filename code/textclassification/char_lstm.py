import re
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
import numpy as np

"""
基于LSTM和字符级的中文文本分类
"""

maxlen = 300  # 最长序列长度
batchSize = 64


def textToChars(filePath):
    """
    读取文本文件并进行处理
    :param filePath:文件路径
    :return:
    """
    lines = []
    df = pd.read_excel(filePath)
    for index, row in df.iterrows():
        row = re.sub("[^\u4e00-\u9fa5]", "", str(row))  # 只保留中文
        lines.append(list(row))
    return lines


def getWordIndex(vocabPath):
    """
    获取word2Index,index2Word
    :param vocabPath:词汇文件
    :return:
    """
    word2Index = {}
    with open(vocabPath, encoding="utf-8") as f:
        for line in f.readlines():
            word2Index[line.strip()] = len(word2Index)
    index2Word = dict(zip(word2Index.values(), word2Index.keys()))
    return word2Index, index2Word


def lodaData(posFile, negFile, word2Index):
    """
    获取训练数据
    :param posFile:正样本文件
    :param negFile:负样本文件
    :param word2Index:
    :return:
    """
    lens = []
    posLines = textToChars(posFile)
    negLines = textToChars(negFile)
    posIndexLines = [[word2Index[word] if word2Index.get(word) else 0 for word in line] for line in posLines]
    negIndexLines = [[word2Index[word] if word2Index.get(word) else 0 for word in line] for line in negLines]
    lines = posIndexLines + negIndexLines
    # lens = [len(line) for line in lines]
    labels = [1] * len(posIndexLines) + [0] * len(negIndexLines)
    padSequences = sequence.pad_sequences(lines, maxlen=maxlen, padding="post", truncating="post")

    return padSequences, np.array(labels)


def train(trainX, labels, maxFeatures):
    """
    构建网络及训练模型
    :param trainX: 特征
    :param labels: 标签
    :param maxFeatures:词汇数量
    :return:
    """
    model = Sequential()
    model.add(Embedding(maxFeatures, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation="sigmoid"))
    # 模型编译
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    print("模型开始训练......")
    model.fit(trainX, labels, batch_size=batchSize, epochs=15, validation_split=0.1)
    return model


if __name__ == '__main__':
    posFilePath = "../../data/data2/pos.xls"
    negFilePath = "../../data/data2/neg.xls"
    # lines = textToChars(filePath)
    # print(lines[0])
    vocabPath = "../../data/vocab.txt"
    word2Index, index2Word = getWordIndex(vocabPath)
    # # print(word2Index)
    padSequences, labels = lodaData(posFilePath, negFilePath, word2Index)
    print(padSequences.shape, labels.shape)
    model = train(padSequences, labels, maxFeatures=len(word2Index))
