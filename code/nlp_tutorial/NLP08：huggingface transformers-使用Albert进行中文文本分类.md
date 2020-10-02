# 1.Albert简介

Alber相对于原始BERT模型主要有三点改进：

* embedding 层参数因式分解

* 跨层参数共享

* 将 NSP 任务改为 SOP 任务



## 1.1 embedding 层参数因式分解（Factorized Embedding Parameterization）

原始的 BERT 模型以及各种依据 Transformer 的预训连语言模型都有一个共同特点，即 $E=H$，其中 E 指的是 Embedding Dimension，$H$ 指的是 Hidden Dimension。这就会导致一个问题，当提升 Hidden Dimension 时，Embedding Dimension 也需要提升，最终会导致参数量呈平方级的增加。所以 ALBERT 的作者将 $E 和 H$ 进行解绑**，具体的操作就是**在 Embedding 后面加入一个矩阵进行维度变换。$E$ 的维度是不变的，如果 $H$ 增大了，我们只需要在 $E$ 后面进行一个升维操作即可

![](https://s1.ax1x.com/2020/08/18/ducEEq.png#shadow)

原本参数数量为 $V∗H$，V 表示的是 Vocab Size。分解成两步则减少为$ V∗E+E∗H$，当$ H$ 的值很大时，这样的做法能够大幅降低参数数量

> V∗H=30000∗768=23,040,000
>
> V∗E+E∗H=30000∗256+256∗768=7,876,608



通过因式分解 Embedding 的实验可以看出，对于参数不共享的版本，随着 $E$ 的增大，效果是不断提升的。但是参数共享的版本似乎不是这样，$E$ 最大并不是效果最好。同时也能发现参数共享对于效果可能带来 1-2 个点的下降

![img](https://s1.ax1x.com/2020/08/18/du74L6.png#shadow)



## 1.2 跨层参数共享（Cross-Layer Parameter Sharing）

传统 Transformer 的每一层参数都是独立的，包括各层的 self-attention、全连接。这样就导致层数增加时，参数量也会明显上升。之前有工作试过单独将 self-attention 或者全连接层进行共享，都取得了一些效果。ALBERT 作者尝试将所有层的参数进行共享，相当于只学习第一层的参数，并在剩下的所有层中重用该层的参数，而不是每个层都学习不同的参数

![img](https://s1.ax1x.com/2020/08/18/duRSC6.png#shadow)

同时作者通过实验还发现了，使用参数共享可以有效的提升模型稳定性，实验结果如下图

![img](https://s1.ax1x.com/2020/08/18/duW7m4.png#shadow)

BERT-base 和 ALBERT 使用相同的层数以及 768 个隐藏单元，结果 BERT-base 共有 1.1 亿个参数，而 ALBERT 只有 3100 万个参数。通过实验发现，feed-forward 层的参数共享会对精度产生比较大的影响；共享注意力参数的影响是最小的

![img](https://s1.ax1x.com/2020/08/18/duhvLD.png#shadow)



## 1.3 将 NSP 任务改为 SOP 任务（Sentence-Order Prediciton (SOP)）

**BERT** 引入了一个叫做**下一个句子预测**的二分类问题。这是专门为提高使用句子对，如 "自然语言推理" 的下游任务的性能而创建的。但是像 RoBERTa 和 XLNet 这样的论文已经阐明了 NSP 的无效性，并且发现它对下游任务的影响是不可靠的

因此，ALBERT 提出了另一个任务 —— **句子顺序预测**。关键思想是：

- 从同一个文档中取两个连续的句子作为一个正样本
- 交换这两个句子的顺序，并使用它作为一个负样本

![img](https://s1.ax1x.com/2020/08/18/duo7nK.png#shadow)

SOP 提高了下游多种任务（SQUAD，MNLI，SST-2，RACE）的表现

![img](https://s1.ax1x.com/2020/08/18/duT1N4.png#shadow)

# 2.使用Albert进行文本分类

```python
import torch
from transformers import BertTokenizer,BertModel,BertConfig
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
import pandas as pd
```

## 2.1 加载预训练模型

```python
pretrained = 'voidful/albert_chinese_small' #使用small版本Albert
tokenizer = BertTokenizer.from_pretrained(pretrained)
model=BertModel.from_pretrained(pretrained)
config=BertConfig.from_pretrained(pretrained)
```

```python
inputtext = "今天心情情很好啊，买了很多东西，我特别喜欢，终于有了自己喜欢的电子产品，这次总算可以好好学习了"
tokenized_text=tokenizer.encode(inputtext)
input_ids=torch.tensor(tokenized_text).view(-1,len(tokenized_text))
outputs=model(input_ids)
```

输出字向量表示和句向量

```python
outputs[0].shape,outputs[1].shape
```

## 2.2 构建模型网络结构

```python
class AlbertClassfier(torch.nn.Module):
    def __init__(self,bert_model,bert_config,num_class):
        super(AlbertClassfier,self).__init__()
        self.bert_model=bert_model
        self.dropout=torch.nn.Dropout(0.4)
        self.fc1=torch.nn.Linear(bert_config.hidden_size,bert_config.hidden_size)
        self.fc2=torch.nn.Linear(bert_config.hidden_size,num_class)
    def forward(self,token_ids):
        bert_out=self.bert_model(token_ids)[1] #句向量 [batch_size,hidden_size]
        bert_out=self.dropout(bert_out)
        bert_out=self.fc1(bert_out) 
        bert_out=self.dropout(bert_out)
        bert_out=self.fc2(bert_out) #[batch_size,num_class]
        return bert_out
```

```python
albertBertClassifier=AlbertClassfier(model,config,2)
device=torch.device("cuda:0") if torch.cuda.is_available() else 'cpu'
albertBertClassifier=albertBertClassifier.to(device)
```

## 2.3 准备训练数据和验证数据

```python
def get_train_test_data(pos_file_path,neg_file_path,max_length=100,test_size=0.2):
    data=[]
    label=[]
    pos_df=pd.read_excel(pos_file_path,header=None)
    pos_df.columns=['content']
    for index, row in pos_df.iterrows():
        row=row['content']
        ids=tokenizer.encode(row.strip(),max_length=max_length,padding='max_length',truncation=True)
        data.append(ids)
        label.append(1)
        
    neg_df=pd.read_excel(neg_file_path,header=None)
    neg_df.columns=['content']
    for index, row in neg_df.iterrows():
        row=row['content']
        ids=tokenizer.encode(row.strip(),max_length=max_length,padding='max_length',truncation=True)
        data.append(ids)
        label.append(0)
    X_train, X_test, y_train, y_test=train_test_split(data,label,test_size=test_size,shuffle=True)
    return (X_train,y_train),(X_test,y_test)
```

```python
pos_file_path="../input/data01/pos.xls"
neg_file_path="../input/data01/neg.xls"
(X_train,y_train),(X_test,y_test)=get_train_test_data(pos_file_path,neg_file_path)
len(X_train),len(X_test),len(y_train),len(y_test),len(X_train[0])
```

```python
class DataGen(data.Dataset):
    def __init__(self,data,label):
        self.data=data
        self.label=label
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self,index):
        return np.array(self.data[index]),np.array(self.label[index])
```

```python
train_dataset=DataGen(X_train,y_train)
test_dataset=DataGen(X_test,y_test)
train_dataloader=data.DataLoader(train_dataset,batch_size=256)
test_dataloader=data.DataLoader(test_dataset,batch_size=256)
```

## 2.4.定义优化器和损失函数

```python
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(albertBertClassifier.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-4)
```

## 2.5.模型训练和测试

```python
for epoch in range(50):
    loss_sum=0.0
    accu=0
    albertBertClassifier.train()
    for step,(token_ids,label) in enumerate(train_dataloader):
        token_ids=token_ids.to(device)
        label=label.to(device)
        out=albertBertClassifier(token_ids)
        loss=criterion(out,label)
        optimizer.zero_grad()
        loss.backward() #反向传播
        optimizer.step() #梯度更新
        loss_sum+=loss.cpu().data.numpy()
        accu+=(out.argmax(1)==label).sum().cpu().data.numpy()
        
    test_loss_sum=0.0
    test_accu=0
    albertBertClassifier.eval()
    for step,(token_ids,label) in enumerate(test_dataloader):
        token_ids=token_ids.to(device)
        label=label.to(device)
        with torch.no_grad():
            out=albertBertClassifier(token_ids)
            loss=criterion(out,label)
            test_loss_sum+=loss.cpu().data.numpy()
            test_accu+=(out.argmax(1)==label).sum().cpu().data.numpy()
    print("epoch % d,train loss:%f,train acc:%f,test loss:%f,test acc:%f"%(epoch,loss_sum/len(train_dataset),accu/len(train_dataset),test_loss_sum/len(test_dataset),test_accu/len(test_dataset)))   
```

 **参考：**

【1】https://www.wmathor.com/index.php/archives/1480/



代码：https://github.com/chongzicbo/nlp-ml-dl-notes/blob/master/code/textclassification/nlp08_huggingface_transformers_albert.ipynb