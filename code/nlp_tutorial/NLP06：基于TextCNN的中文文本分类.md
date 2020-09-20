# 1.TextCNN基本原理

<img src="https://pic1.zhimg.com/v2-38e6e46009ea88c06465ed0770051c4d_r.jpg" style="zoom: 67%;" />

<img src="https://pic4.zhimg.com/v2-2ea1f0b8b166f31273b26bca3ba8e8b2_r.jpg" style="zoom:67%;" />

主要看第二张图：

* 第一层为输入层，输入是一个$n \times k$的矩阵,图中为$7 \times 5$。其中$n$为句子中的单词数，$k$为词向量维度。词向量可以是预训练好的，也可以在网络中重新开始训练。第一张图中输入有两个矩阵，其中一个使用的预训练好的向量，另一个则作为训练参数。
* 第二层为卷积层，可以把矩阵理解为一张channels为1的图像，使用宽度同词向量维度一样的卷积核去做卷积运算，且卷积核只在高度方向(单词方向).因此每次卷积核滑动的位置都是完整的单词，保证了单词作为语言中最小粒度的合理性。假设词向量维度为embedding_dim,卷积核高度为filter_window_size,则卷积核大小为(embedding_dim,filter_window_size),卷积后的大小为(sequence_len-filter_window_size,1)
* 第三层为卷积层，经过max_pooling后得到一个标量。实际中会使用num_filters卷积核同时卷积，每个卷积核得到的标量拼接在一起形成一个向量。此外，也会使用多个filter_window_size(如图2中3个filter_window_size分别为3、4、5)，每个filter_window_size会得到一个向量，最后把所有的向量拼接在一起，然后接一个softmax进行分类。

# 2. TextCNN实现

```python
import torch
from torch import nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset,DataLoader
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import re
from tensorflow.keras.preprocessing import sequence


maxlen=300
batch_size=128
```

## 2.1 数据预处理

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
    posLines = textToChars(posFile)
    negLines = textToChars(negFile)
    posIndexLines = [[word2Index[word] if word2Index.get(word) else 0 for word in line] for line in posLines]
    negIndexLines = [[word2Index[word] if word2Index.get(word) else 0 for word in line] for line in negLines]
    lines = posIndexLines + negIndexLines
    print("训练样本和测试样本共：%d 个"%(len(lines)))
    # lens = [len(line) for line in lines]
    labels = [1] * len(posIndexLines) + [0] * len(negIndexLines)
    padSequences = sequence.pad_sequences(lines, maxlen=maxlen, padding="post", truncating="post")
    X_train,X_test,y_train,y_test=train_test_split(padSequences,np.array(labels),test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test
```

```python
vocabPath="/content/drive/My Drive/data/vocab.txt"
negFilePath="/content/drive/My Drive/data/text_classify/sentiment/neg.xls"
posFilePath="/content/drive/My Drive/data/text_classify/sentiment/pos.xls"
word2Index, index2Word=getWordIndex(vocabPath)
X_train,X_test,y_train,y_test=lodaData(posFile=posFilePath,negFile=negFilePath,word2Index=word2Index)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
```

```python
class MyDataset(Dataset):

  def __init__(self,features,labels):
    """
    features:文本向量化后的特征
    labels:标签向量 
    """
    self.features=features
    self.labels=labels

  def __len__(self):
    return self.features.shape[0]

  def __getitem__(self,index):
    return self.features[index],self.labels[index]

    
train_dataset=MyDataset(X_train,y_train)
test_dataset=MyDataset(X_test,y_test)
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
```

## 2.2 TextCNN实现

```python
class TextCnn(nn.Module):
    def __init__(self, param: dict):
        super(TextCnn, self).__init__()
        input_channel = 1  # input channel size
        output_channel = param["output_channel"]  # output channel size
        kernel_size = param["kernel_size"]
        vocab_size = param["vocab_size"]
        embedding_dim = param["embedding_dim"]
        dropout = param["dropout"]
        class_num = param["class_num"]
        self.param = param
        self.embedding = nn.Embedding(vocab_size, embedding_dim,padding_idx=0)
        self.conv1 = nn.Conv2d(input_channel, output_channel, (kernel_size[0], embedding_dim))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (kernel_size[1], embedding_dim))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (kernel_size[2], embedding_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * output_channel, class_num)

    def init_embedding(self, embedding_matrix):
        self.embedding.weight = nn.Parameter(torch.Tensor(embedding_matrix))

    @staticmethod
    def conv_pool(x, conv):
        """
        卷积+池化
        :param x:[batch_size,1,sequence_length,embedding_dim]
        :param conv:
        :return:
        """
        x = conv(x)  # 卷积， [batch_size,output_channel,h_out,1]
        x = F.relu((x.squeeze(3)))  # 去掉最后一维,[batch_size,output_channel,h_out]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # [batch_size,output_channel]
        return x

    def forward(self, x):
        """
        前向传播
        :param x:[batch_size,sequence_length]
        :return:
        """
        x = self.embedding(x)  # [batch_size,sequence_length,embedding_dim]
        x = x.unsqueeze(1)  # 增加一个channel维度 [batch_size,1,sequence_length,embedding_dim]
        x1 = self.conv_pool(x, self.conv1)  # [batch_size,output_channel]
        x2 = self.conv_pool(x, self.conv2)  # [batch_size,output_channel]
        x3 = self.conv_pool(x, self.conv3)  # [batch_size,output_channel]
        x = torch.cat((x1, x2, x3), 1)  # [batch_size,output_channel*3]
        x = self.dropout(x)
        logit = F.log_softmax(self.fc1(x), dim=1)
        return logit

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

```

## 2.3 模型训练

```python
textCNNParams={
    "vocab_size":len(word2Index),
    "embedding_dim":100,
    "class_num":2,
    "output_channel":4,
    "kernel_size":[3,4,5],
    "dropout":0.2
}
```

```python
net=TextCnn(textCNNParams)
# net.init_weights()

net.cuda()
```

```python
optimizer=torch.optim.SGD(net.parameters(),lr=0.01)
criterion=nn.NLLLoss()
```

```python
for epoch in range(10):
  total_train_loss=[]
  net.train()
  for i,(feature,label) in enumerate(train_dataloader):
    feature=feature.cuda()
    label=label.cuda()
    y_pred=net(feature.long()) #前向计算
    loss=criterion(y_pred,label) #计算损失
    optimizer.zero_grad() #清除梯度
    loss.backward() #计算梯度，误差回传
    optimizer.step() #更新参数
    total_train_loss.append(loss.data.item())
  total_valid_loss=[]
  pred_true_labels=0
  net.eval()
  for i,(feature_test,label_test) in enumerate(test_dataloader):
    feature_test=feature_test.cuda()
    label_test=label_test.cuda()
    with torch.no_grad():
      pred_test=net(feature_test.long())
      test_loss=criterion(pred_test,label_test)
      total_valid_loss.append(test_loss.data.item())
      # accu=torch.sum((torch.argmax(pred_test,dim=1)==label_test)).data.item()/feature_test.shape[0]
      pred_true_labels+=torch.sum(torch.argmax(pred_test,dim=1)==label_test).data.item()
      
  print("epoch:{},train_loss:{},test_loss:{},accuracy:{}".format(epoch,np.mean(total_train_loss),np.mean(total_valid_loss),pred_true_labels/len(test_dataset)))
```

## 2.4 模型测试

```python
def predict_one(sentence,net,word2Index):
  sentence=re.sub("[^\u4e00-\u9fa5]", "", str(sentence))  # 只保留中文
  print(sentence)
  sentence=[word2Index[word] if word2Index.get(word) else 0 for word in sentence]
  sentence=sentence+[0]*(maxlen-len(sentence)) if len(sentence)<maxlen else sentence[0:300]
  print(sentence)
  sentence=torch.tensor(np.array(sentence)).view(-1,len(sentence)).cuda()
  label=torch.argmax(net(sentence),dim=1).data.item()
  print(label)
	
```

```py
sentence="一次很不爽的购物，页面上说是第二天能到货，结果货是从陕西发出的，卖家完全知道第二天根本到不了货。多处提到送货入户还有100%送货入户也没有兑现，与客服联系多日，还是把皮球踢到快递公司。算是一个教训吧。"
predict_one(sentence,net,word2Index)
```

代码：https://github.com/chongzicbo/nlp-ml-dl-notes/blob/master/code/textclassification/text_cnn.ipynb