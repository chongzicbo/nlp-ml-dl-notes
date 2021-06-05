<center><b><font color=#A52A2A size=5 >公众号：数据挖掘与机器学习笔记</font></b></center>



在之前的文章“[论文阅读04：使用序列标注的方式解决实体和关系的联合抽取](https://blog.csdn.net/u013230189/article/details/109953955)”介绍了一种使用序列标注的方式来进行实体和关系的联合抽取，模型的具体细节可以查看该文章。今天的文章主要是对这篇文章中论文提到的模型进行简单的实现。论文中提到的偏置目标函数通过给不同的标签赋予不同的权重进行实现。

这里仅实现模型，能够跑通，因为是在个人笔记本上跑，机器性能不够，所以没有训练完，也没有调参。最终的训练效果也未知。感兴趣的同学可以自己调调参。

# 1.数据处理

```python
import re
import json
import numpy as np
from tensorflow.keras.preprocessing import sequence
from config import *


def get_data(train_path, test_path):
    """
    从json中提取数据
    :param train_path:
    :param test_path:
    :return:
    """
    train_file = open(train_path).readlines()
    x_train = []
    y_train = []
    for i in train_file:
        data = json.loads(i)
        x_data, y_data = data_decoding(data)
        x_train += x_data
        y_train += y_data
    test_file = open(test_path).readlines()
    x_test = []
    y_test = []
    for j in test_file:
        data = json.loads(j)
        x_data, y_data = data_decoding(data)
        x_test += x_data
        y_test += y_data
    return x_train, y_train, x_test, y_test


def data_decoding(data):
    '''
      decode the json file
      sentText is the sentence
      each sentence may have multiple types of relations
      for every single data, it contains: (sentence-splited, labels)
    '''
    sentence = data["sentText"]
    relations = data["relationMentions"]
    x_data = []
    y_data = []
    for i in relations:
        entity_1 = i["em1Text"].split(" ")
        entity_2 = i["em2Text"].split(" ")
        relation = i["label"]
        relation_label_1 = entity_label_construction(entity_1)
        relation_label_2 = entity_label_construction(entity_2)
        output_list = sentence_label_construction(sentence, relation_label_1, relation_label_2, relation)
        x_data.append(sentence.split(" "))
        y_data.append(output_list)
    return x_data, y_data


def entity_label_construction(entity):
    '''
        give each word in an entity the label
        for entity with multiple words, it should follow the BIES rule
    '''
    relation_label = {}
    for i in range(len(entity)):
        if i == 0 and len(entity) >= 1:
            relation_label[entity[i]] = "B"
        if i != 0 and len(entity) >= 1 and i != len(entity) - 1:
            relation_label[entity[i]] = "I"

        if i == len(entity) - 1 and len(entity) >= 1:
            relation_label[entity[i]] = "E"

        if i == 0 and len(entity) == 1:
            relation_label[entity[i]] = "S"
    return relation_label


def sentence_label_construction(sentence, relation_label_1, relation_label_2, relation):
    '''
       combine the label for each word in each entity with the relation
       and then combine the relation-entity label with the position of the entity in the triple
    '''
    element_list = sentence.split(" ")
    dlist_1 = list(relation_label_1)
    dlist_2 = list(relation_label_2)
    output_list = []
    for i in element_list:
        if i in dlist_1:
            output_list.append(relation + "-" + relation_label_1[i] + "-1")
        elif i in dlist_2:
            output_list.append(relation + "-" + relation_label_2[i] + "-1")
        else:
            output_list.append("O")
    return output_list


def format_control(string):
    str1 = re.sub(r"\r", "", string)
    str2 = re.sub(r"\n", "", str1)
    str3 = re.sub(r"\s*", "", str2)
    return str3


def get_index(word_dict, tag_dict, x_data, y_data):
    x_out = [word_dict[str(k)] for k in x_data]
    y_out = [tag_dict.get(str(l), tag_dict["O"]) for l in y_data]
    return [x_out, y_out]


def word_tag_dict(word_dict_path, tag_dict_path):
    word_dict = {}
    f = open(word_dict_path, "r").readlines()
    for i, j in enumerate(f):
        word = re.sub(r"\n", "", str(j))
        word_dict[word] = i + 1
    tag_dict = {}
    f = open(tag_dict_path, "r").readlines()
    for m, n in enumerate(f):
        tag = re.sub(r"\n", "", str(n))
        tag_dict[tag] = m
    return word_dict, tag_dict


class DataGenerator:
    def __init__(self, word_dict, tag_dict, x_train, y_train, batch_size, max_len, is_test=False):
        self.max_len = max_len
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.is_test = is_test
        self.steps = len(self.x_train) // self.batch_size
        if len(self.x_train) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.x_train)))
            if not self.is_test:
                np.random.shuffle(idxs)
            x_data, y_data = [], []
            for i in idxs:
                x = self.x_train[i]
                y = self.y_train[i]
                x_out, y_out = get_index(self.word_dict, self.tag_dict, x, y)
                x_data.append(x_out)
                y_data.append(y_out)
                if len(x_data) == self.batch_size or i == idxs[-1]:
                    x_data = sequence.pad_sequences(x_data, maxlen=self.max_len, padding="post", truncating="post")
                    y_data = sequence.pad_sequences(y_data, maxlen=self.max_len, padding="post", truncating="post", value=self.tag_dict["O"])
                    yield np.array(x_data), np.array(y_data)
                    x_data, y_data = [], []


if __name__ == '__main__':
    sentence_train, seq_train, sentence_test, seq_test = get_data(TRAIN_PATH, TEST_PATH)
    max_len = max([len(s) for s in sentence_train])
    word_dict, tag_dict = word_tag_dict(WORD_DICT, TAG_DICT)
    dataGenerator = DataGenerator(word_dict, tag_dict, sentence_train, seq_train, 16, max_len)
    for x, y in dataGenerator.__iter__():
        print(x.shape, y.shape)
        print(x[0])
        print(y[0])
        break

```

# 2. 模型实现

```python
from tensorflow import keras
from tensorflow.keras import layers
from config import MAX_LEN, BATCH_SIZE, LSTM_DECODE, LSTM_ENCODE, WORD_DICT, TAG_DICT, EMBEDDING_SIZE


def bilstm_lstm(word_size, tag_size):
    x = layers.Input(shape=MAX_LEN, batch_size=BATCH_SIZE)
    embedding_x = layers.Embedding(input_dim=word_size, output_dim=EMBEDDING_SIZE)(x)
    bilstm_encode = layers.Bidirectional(layers.LSTM(units=LSTM_ENCODE, return_sequences=True))(embedding_x)
    bilstm_decode = layers.LSTM(units=LSTM_DECODE, return_sequences=True)(bilstm_encode)
    out = layers.Dense(units=tag_size, activation="softmax")(bilstm_decode)
    model = keras.models.Model(x, out)
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    model.summary()
    return model

```

# 3.模型训练

```python
from data_helper import get_data, word_tag_dict, DataGenerator
from config import MAX_LEN, TRAIN_PATH, TEST_PATH, WORD_DICT, TAG_DICT, EPOCH_NUM, BATCH_SIZE, model_save_path
from model import bilstm_lstm


def get_class_weight(tag_dict):
    """
        不同标签的权重不一样，标签"O"的权重为1，其他的为10
    :param tag_dict:
    :return:
    """
    class_weight = {}
    for tag, index in tag_dict.items():
        if tag == "O":
            class_weight[index] = 1
        else:
            class_weight[index] = 10
    return class_weight


def train_bilstm_lstm():
    x_train, y_train, x_test, y_test = get_data(TRAIN_PATH, TEST_PATH)
    word_dict, tag_dict = word_tag_dict(WORD_DICT, TAG_DICT)
    train_dataGenerator = DataGenerator(word_dict, tag_dict, x_train, y_train, BATCH_SIZE, MAX_LEN)
    test_dataGenerator = DataGenerator(word_dict, tag_dict, x_test, y_test, BATCH_SIZE, MAX_LEN)
    class_weight = get_class_weight(tag_dict)
    print(class_weight)
    model = bilstm_lstm(len(word_dict) + 1, len(tag_dict))
    model.fit_generator(train_dataGenerator.__iter__(), epochs=EPOCH_NUM, steps_per_epoch=train_dataGenerator.steps,
                        validation_data=test_dataGenerator, validation_steps=test_dataGenerator.steps, class_weight=class_weight)
    model.save_weights(filepath=model_save_path)


if __name__ == '__main__':
    train_bilstm_lstm()
```



所用训练数据较大，未上传至github，需要请私信。

代码：[https://github.com/chongzicbo/KG_Tutorial/tree/main/relation_extract/joint_re_bilstm_ntc](https://github.com/chongzicbo/KG_Tutorial/tree/main/relation_extract/joint_re_bilstm_ntc)



# 参考：

[1]https://aistudio.baidu.com/aistudio/projectdetail/26338

[2]https://github.com/gswycf/Joint-Extraction-of-Entities-and-Relations-Based-on-a-Novel-Tagging-Scheme/blob/master/train.py

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200828221113544.jpg#pic_center)