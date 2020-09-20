使用CNN-LSTM进行情感分类，这里是一个二分类模型。整体上分为以下几个步骤：

* 环境及参数设置
* 数据预处理
* 模型网络结构搭建及训练
* 模型使用

# 1. 环境及参数设置

环境主要指需要哪些包，参数设置包括Embedding、CNN、LSTM网络层的参数和一些基本参数设置。

```python
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Embedding,LSTM,Conv1D,MaxPooling1D
from tensorflow.keras.datasets import imdb
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```



```python
#embedding 参数
maxlen=100 #最大样本长度，不足进行Padding，超过进行截取
embedding_size=200 #词向量维度

#卷积参数
kernel_size=5
filters=128
pool_size=4

#LSTM参数
lstm_output_size=100 #LSTM层的输出维度

#训练参数
batch_size=128
epochs=20
```



# 2. 数据预处理及训练数据准备

## 2.1 数据概览

这里使用的情感分类数据主要是一些购物、酒店住宿等评论。使用Excel的格式提供，每一行表示一个样本。具体形式、内容如下图：

![image-20200908164538892](http://qfth8dccq.hn-bkt.clouddn.com/images/image-20200908164538892.png)

<center>负面评论</center>



![image-20200908164816921](http://qfth8dccq.hn-bkt.clouddn.com/images/image-20200908164816921.png)

<center>正面评论</center>

## 2.2 数据预处理

这里仅做了比较简单的文本处理：只保留中文字符，去掉所有非中文字符。另外，没有进行分词，使用字符级的模型进行训练。

```python
def textToChars(filePath):
  """
  读取文本文件并进行处理
  :param filePath:文件路径
  :return:
  """
  lines = []
  df=pd.read_excel(filePath,header=None)
  df.columns=['content']
  for index, row in df.iterrows():
    row=row['content']
    row = re.sub("[^\u4e00-\u9fa5]", "", str(row))  # 只保留中文
    lines.append(list(str(row)))
  return lines
```

## 2.3 训练数据准备

将文本数据转换成训练所需的矩阵格式，并划分训练集和测试集。

```python
def getWordIndex(vocabPath):
  """
  获取word2Index,index2Word
  :param vocabPath:词汇文件，使用的是BERT里的vocab.txt文件
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
  posLines = textToChars(posFile)
  negLines = textToChars(negFile)
  textLines=posLines+negLines
  print("正样本数量%d,负样本数量%d"%(len(posLines),len(negLines)))
  posIndexLines = [[word2Index[word] if word2Index.get(word) else 0 for word in line] for line in posLines]
  negIndexLines = [[word2Index[word] if word2Index.get(word) else 0 for word in line] for line in negLines]
  lines = posIndexLines + negIndexLines
  print("训练样本和测试样本共：%d 个"%(len(lines)))
  # lens = [len(line) for line in lines]
  labels = [1] * len(posIndexLines) + [0] * len(negIndexLines)
  padSequences = sequence.pad_sequences(lines, maxlen=maxlen, padding="post", truncating="post")
  X_train,X_test,y_train,y_test=train_test_split(padSequences,np.array(labels),test_size=0.2,random_state=42) #按照8:2的比例划分训练集和测试集
  return (textLines,labels),(X_train,X_test,y_train,y_test)
```

```python
vocabPath="/content/drive/My Drive/data/vocab.txt"
negFilePath="/content/drive/My Drive/data/text_classify/sentiment/neg.xls"
posFilePath="/content/drive/My Drive/data/text_classify/sentiment/pos.xls"
word2Index, index2Word=getWordIndex(vocabPath)
(textLines,labels),(X_train,X_test,y_train,y_test)=lodaData(posFile=posFilePath,negFile=negFilePath,word2Index=word2Index)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
```

![image-20200908165302338](http://qfth8dccq.hn-bkt.clouddn.com/images/image-20200908165302338.png)

样本总数为21005，正负样本数量大致持平。

# 3. 模型网络结构搭建及训练

整体的网络结构为：Embedding+Conv+LSTM+Dense,其中，卷积层是一维卷积，在时间步上进行卷积。Embedding之后要进行Dropout,卷积之后需要进行MaxPooling，最后的全连接层后要接一个sigmoid激活函数。损失函数使用二分类的交叉熵损失，优化器使用Adam。

```python
model=Sequential()
model.add(Embedding(len(word2Index),embedding_size,input_length=maxlen))
model.add(Dropout(0.2))
model.add(Conv1D(filters,kernel_size,padding="valid",activation="relu",strides=1))
model.add(MaxPooling1D(pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("开始训练")
model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test,y_test))
```

```python
model.summary()
```

![image-20200908170258793](http://qfth8dccq.hn-bkt.clouddn.com/images/image-20200908170258793.png)

# 4.模型使用



```python
def predict_one(sentence,model,word2Index):
  sentence=re.sub("[^\u4e00-\u9fa5]", "", str(sentence))  # 只保留中文
  # print(sentence)
  sentence=[word2Index[word] if word2Index.get(word) else 0 for word in sentence]
  sentence=sentence+[0]*(maxlen-len(sentence)) if len(sentence)<maxlen else sentence[0:300]
  # print(sentence)
  sentence=np.reshape(np.array(sentence),(-1,len(sentence))) 
  pred_prob=model.predict(sentence)
  label = 1 if pred_prob[0][0]>0.5 else 0
  print(label)
  return label
```

```python
sentence="一次很不爽的购物，页面上说是第二天能到货，结果货是从陕西发出的，卖家完全知道第二天根本到不了货。多处提到送货入户还有100%送货入户也没有兑现，与客服联系多日，还是把皮球踢到快递公司。算是一个教训吧。"
predict_one(sentence,model,word2Index)
```



* 数据：https://github.com/chongzicbo/nlp-ml-dl-notes/tree/master/data/data2

* 代码：https://github.com/chongzicbo/nlp-ml-dl-notes/blob/master/code/textclassification/cnn_lstm.ipynb

