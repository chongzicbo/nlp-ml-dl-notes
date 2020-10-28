import os
import jieba
import random
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 参数设置
word_size = 64  # 词向量维度
window = 5  # 窗口大小
nb_negative = 25  # 随机负采样的样本数
min_count = 10  # 频数少于min_count的词会将被抛弃，低频词类似于噪声，可以抛弃掉
file_num = 1000


# 数据预处理
def get_all_apths(dirname):
    paths = []  # 将所有的txt文件路径存放在这个list中
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)  # 合并成一个完整路径
            paths.append(apath)
    return paths


def get_corpus(file_path):
    words = []
    corpus = []
    i = 0
    for file in file_path:
        if ".txt" in file:
            i += 1
            try:
                with open(file, encoding="utf-8") as fr:
                    for line in fr:
                        words += jieba.lcut(line)
                        corpus.append(jieba.lcut(line))
            except Exception as e:
                print(e)
        if i == file_num:
            break

    words = dict(Counter(words))
    total = sum(words.values())
    words = {i: j for i, j in words.items() if j >= min_count}  # 去掉低频词
    id2word = {i + 2: j for i, j in enumerate(words)}
    id2word[0] = "PAD"
    id2word[1] = "UNK"
    word2id = {j: i for i, j in id2word.items()}
    return words, corpus, id2word, word2id


def get_negative_sample(x, word_range, neg_num):
    """
    负采样
    :param x:
    :param word_range:
    :param neg_num:
    :return:
    """
    negs = []
    while True:
        rand = random.randrange(0, word_range)
        if rand not in negs and rand != x:
            negs.append(rand)
        if len(negs) == neg_num:
            return negs


def data_generator(corpus, word2id, id2word):
    """
    生成训练数据
    :return:
    """
    x, y = [], []
    for sentence in corpus:
        sentence = [0] * window + [word2id[w] for w in sentence if w in word2id] + [0] * window
        # 上面这句代码的意思是，因为我们是通过滑窗的方式来获取训练数据的，那么每一句语料的第一个词和最后一个词
        # 如何出现在中心位置呢？答案就是给它padding一下，例如“我/喜欢/足球”，两边分别补窗口大小个pad，得到“pad pad 我 喜欢 足球 pad pad”
        # 那么第一条训练数据的背景词就是['pad', 'pad','喜欢', '足球']，中心词就是'我'
        for i in range(window, len(sentence) - window):
            x.append(sentence[i - window:i] + sentence[i + 1:window + i + 1])
            y.append([sentence[i]] + get_negative_sample(sentence[i], len(id2word), nb_negative))
    x, y = np.array(x), np.array(y)
    z = np.zeros((len(x), nb_negative + 1))
    z[:, 0] = 1
    return x, y, z


def get_train_test_data(x, y, z):
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split([x, y, z], test_size=0.2, random_state=42, shuffle=True)
    return X_train, X_test, y_train, y_test, z_train, z_test


class DatasetTorch(Dataset):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z[:, 1]  # torch使用交叉熵损失时，target不需要使用onehot

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]


def get_train_test_dataloader(x, y, z, batch_size):
    """
    生成训练和测试数据的DataLoader
    :param x:
    :param y:
    :param z:
    :param batch_size:
    :return:
    """
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x, y, z, test_size=0.2, random_state=42, shuffle=True)
    train_dataset = DatasetTorch(x_train, y_train, z_train)
    test_dataset = DatasetTorch(x_test, y_test, z_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataloader, test_dataloader
