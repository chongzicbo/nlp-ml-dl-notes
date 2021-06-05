<center><b><font color=#A52A2A size=5 >公众号：数据挖掘与机器学习笔记</font></b></center>

《Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification》是2016年由中国科学技术大学Peng Zhou等在ACL发表的论文，本文是对该论文的简单解读和代码复现。

# 1.模型结构

![image-20201107121353543](https://gitee.com/chengbo123/images/raw/master/image-20201107121353543.png)

现在来看，模型的结构还是比较简单的，主要包括5部分，分别是输入层、词嵌入层、BiLSTM层、Attention层和softmax输出层。

## 1.1 输入层

输入的是句子，可以是字符序列也可以是单词序列，或者两者相结合。此外，对于句子中的两个实体，分别计算各个字符相对于实体的位置。比如有如下样本:

"但李世民回来之后，李渊又反悔，听信后妃们的求情，保留了李建成的太子之位。"

在这个样本中，实体1为李世民，实体2为李建成，关系为兄弟姐妹。对于第一个字符“但”字来说，其相对于实体1的距离为(但字在字符序列中的索引-实体1在字符序列中的索引)，相对于实体2的距离为（但字在字符序列中的索引-实体2在字符序列中的索引）。因此模型的输入为字符序列+字符的相对位置编码。

## 1.2 Embeddings

模型的嵌入层包括字符或词嵌入以及相对位置编码的嵌入。字符嵌入可以随机初始化一个也可使使用预训练好的向量。具体的可以参考后面的代码。

## 1.3 BiLSTM层

 双向LSTM是RNN的一种改进，其主要包括前后向传播，每个时间点包含一个LSTM单元用来选择性的记忆、遗忘和输出信息。LSTM单元的公式如下：

![image-20201107134150592](https://gitee.com/chengbo123/images/raw/master/image-20201107134150592.png)

对输入进行前向和后向遍历，然后将结果加和。

![image-20201107134920060](https://gitee.com/chengbo123/images/raw/master/image-20201107134920060.png)



## 1.4  Attention层

![image-20201107134948231](C:\Users\36184\AppData\Roaming\Typora\typora-user-images\image-20201107134948231.png)

中，$H$是BiLSTM的输出矩阵$[h_1,h_2,\ldots,h_T]$，T是序列长度。$H\in R^{d^w \times T}$,$d^w$是词向量的维度。$w$是可训练的向量，$w^T$是其转置。$w,\alpha,r$的维度分别是$d^w,T,d^w$

最后得到

![image-20201107135622441](https://gitee.com/chengbo123/images/raw/master/image-20201107135622441.png)

用于softmax分类。

![image-20201107135747042](https://gitee.com/chengbo123/images/raw/master/image-20201107135747042.png)

损失函数为：

![image-20201107135805782](https://gitee.com/chengbo123/images/raw/master/image-20201107135805782.png)

# 2.模型结果

![image-20201107135950260](https://gitee.com/chengbo123/images/raw/master/image-20201107135950260.png)

# 3.模型实现

## 3.1 数据处理

```python
import codecs
import pandas as pd
import numpy as np
import collections
from config import *


def get_relation2id(file_path):
    relation2id = {}
    with codecs.open(file_path, "r", "utf-8") as f:
        for line in f.readlines():
            relation2id[line.split()[0]] = int(line.split()[1])
        f.close()
    return relation2id


def get_sentence_label_positionE(file_path, relation2id):
    datas = []
    labels = []
    positionE1 = []
    positionE2 = []
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    with codecs.open(file_path, "r", "utf-8") as f:
        for line in f:
            line_split = line.split("\t")
            # if count[relation2id.get(line[2], 0)] < 1500:
            sentence = []
            index1 = line_split[3].index(line_split[0])  # 实体1在语料中的索引位置
            position1 = []
            index2 = line_split[3].index(line_split[1])  # 实体2在语料中的索引位置
            position2 = []
            for i, word in enumerate(line_split[3]):
                sentence.append(word)
                position1.append(i - index1)  # 字符与实体1的相对位置
                position2.append(i - index2)  # 字符与实体2的相对位置
                i += 1
            datas.append(sentence)
            labels.append(relation2id[line_split[2]])  # 语料对应的标签
            positionE1.append(position1)
            positionE2.append(position2)
            count[relation2id[line_split[2]]] += 1
    return datas, labels, positionE1, positionE2


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def get_word2id(datas):
    all_words = flatten(datas)
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words) + 1)
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    word2id["BLANK"] = len(word2id) + 1
    word2id["UNKNOWN"] = len(word2id + 1)
    id2word[len(id2word) + 1] = "BLANK"
    id2word[len(id2word) + 1] = "UNKNOWN"
    return word2id, id2word


def get_data_array(word2id, datas, labels, positionE1, positionE2, max_len=50):
    def X_padding(words):
        ids = []
        for i in words:
            if i in word2id:
                ids.append(word2id[i])
            else:
                ids.append(word2id["UNKNOWN"])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([word2id["BLANK"]] * (max_len - len(ids)))
        return ids

    def pos(num):
        if num < -40:
            return 0
        if num >= -40 and num <= 40:
            return num + 40
        if num > 40:
            return 80

    def position_padding(words):
        words = [pos(i) for i in words]
        if len(words) >= max_len:
            return words[:max_len]
        words.extend([81] * (max_len - len(words)))
        return words

    df_data = pd.DataFrame({'words': datas, 'tags': labels, 'positionE1': positionE1, 'positionE2': positionE2}, index=range(len(datas)))  # if __name__ == '__main__':
    df_data["words"] = df_data["words"].apply(X_padding)
    df_data["tags"] = df_data["tags"]

    df_data["positionE1"] = df_data["positionE1"].apply(position_padding)
    df_data["positionE2"] = df_data["positionE2"].apply(position_padding)
    datas = np.asarray(list(df_data["words"].values))
    labels = np.asarray(list(df_data["tags"].values))
    positionE1 = np.asarray(list(df_data["positionE1"].values))
    positionE2 = np.asarray(list(df_data["positionE2"].values))
    return datas, labels, positionE1, positionE2
```

## 3.2 模型结构搭建

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model

"""
参考论文
Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
"""


class BiLSTMAttention(Model):
    def __init__(self, config: dict):
        super(BiLSTMAttention, self).__init__()
        self.batch = config["BATCH"]
        self.embedding_size = config["EMBEDDING_SIZE"]
        self.embedding_dim = config["EMBEDDING_DIM"]
        self.hidden_dim = config["HIDDEN_DIM"]
        self.tag_size = config["TAG_SIZE"]
        self.pos_size = config["POS_SIZE"]
        self.pos_dim = config["POS_DIM"]

        self.word_embeds = Embedding(self.embedding_size, self.embedding_dim)
        self.pos1_embeds = Embedding(self.pos_size, self.pos_dim)
        self.pos2_embeds = Embedding(self.pos_size, self.pos_dim)
        self.bilstm = Bidirectional(LSTM(self.hidden_dim // 2, return_sequences=True))
        self.dense = Dense(self.tag_size, activation="softmax")
        self.dropout_lstm = Dropout(0.5)
        self.drop_att = Dropout(0.5)
        self.att_weight = tf.Variable(tf.random.normal(shape=(self.batch, 1, self.hidden_dim)))
        self.relation_bias = tf.Variable(tf.random.normal(shape=(self.batch, self.tag_size, 1)))

    def attention(self, H):
        M = tf.tanh(H)
        a = tf.nn.softmax(tf.matmul(self.att_weight, M), 2)
        a = tf.transpose(a, perm=[0, 2, 1])
        return tf.matmul(H, a)

    def call(self, inputs, training=True):
        embeds = tf.concat((self.word_embeds(inputs[0]), self.pos1_embeds(inputs[1]),
                            self.pos2_embeds(inputs[2])), axis=2)
        # print("embeds shape:", embeds.shape)
        bilstm_out = self.bilstm(embeds)
        # print("lstm_out shape:", bilstm_out.shape)
        if training:
            bilstm_out = self.dropout_lstm(bilstm_out)
        bilstm_out = tf.transpose(bilstm_out, perm=[0, 2, 1])
        # print("transpose lstm_out shape:", bilstm_out.shape)
        att_out = tf.tanh(self.attention(bilstm_out))
        # print("attn_out:", att_out.shape)
        if training:
            att_out = self.drop_att(att_out)
        res = self.dense(tf.squeeze(att_out))
        # print("res shape", res.shape)
        return res


if __name__ == '__main__':
    EMBEDDING_SIZE = 100
    EMBEDDING_DIM = 100

    POS_SIZE = 82  # 不同数据集这里可能会报错。
    POS_DIM = 25

    HIDDEN_DIM = 200

    TAG_SIZE = 12

    BATCH = 128
    EPOCHS = 100

    config = {}
    config['EMBEDDING_SIZE'] = EMBEDDING_SIZE
    config['EMBEDDING_DIM'] = EMBEDDING_DIM
    config['POS_SIZE'] = POS_SIZE
    config['POS_DIM'] = POS_DIM
    config['HIDDEN_DIM'] = HIDDEN_DIM
    config['TAG_SIZE'] = TAG_SIZE
    config['BATCH'] = BATCH
    config["pretrained"] = False

    learning_rate = 0.0005
    model = BiLSTMAttention(config)
    sentence = tf.ones(shape=(BATCH, 50), dtype=tf.int32)
    pos1 = tf.ones(shape=(BATCH, 50), dtype=tf.int32)
    pos2 = tf.ones(shape=(BATCH, 50), dtype=tf.int32)
    model([sentence, pos1, pos2])
    model.summary()
```

## 3.3 模型训练

```
from config import datas, labels, positionE1, positionE2, config, EPOCHS
from bilstm_attention_tf import BiLSTMAttention


def train():
    model = BiLSTMAttention(config)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x=[datas, positionE1, positionE2], y=labels, batch_size=config["BATCH"], epochs=EPOCHS, validation_split=0.2)
    model.summary()
    return history


if __name__ == '__main__':
    train()
```



代码：https://github.com/chongzicbo/KG_Tutorial/tree/main/relation_extract

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200828221113544.jpg#pic_center)

参考：

[1] 《Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification》

[2] https://blog.csdn.net/qq_36426650/article/details/88207917

[3]https://github.com/buppt/ChineseNRE