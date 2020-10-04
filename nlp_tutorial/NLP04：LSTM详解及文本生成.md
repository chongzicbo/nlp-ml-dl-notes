# 1.长短期记忆

LSTM 中引入了3个门，即输入门（input gate）、遗忘门（forget gate）和输出门（output gate），以及与隐藏状态形状相同的记忆细胞（某些文献把记忆细胞当成一种特殊的隐藏状态），从而记录额外的信息。

## 1.1. 输入门、遗忘门和输出门

与门控循环单元中的重置门和更新门一样，如图所示，长短期记忆的门的输入均为当前时间步输入 $X_t$ 与上一时间步隐藏状态 $H_{t−1}$ ，输出由激活函数为sigmoid函数的全连接层计算得到。如此一来，这3个门元素的值域均为 [0,1] 。

<img src="https://zh.gluon.ai/_images/lstm_0.svg" width="500"/>

<center> 长短期记忆中输入门、遗忘门和输出门的计算</center>

具体来说，假设隐藏单元个数为 $h$ ，给定时间步 $t$ 的小批量输入$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$（样本数为 $n$ ，输入个数为 $d$ ）和上一时间步隐藏状态$\boldsymbol{H}_{t-1} \in \mathbb{R}^{n \times h}$。 时间步 $t$ 的输入门$\boldsymbol{I}_t \in \mathbb{R}^{n \times h}$、遗忘门$\boldsymbol{F}_t \in \mathbb{R}^{n \times h}$和输出门$ \boldsymbol{O}_t \in \mathbb{R}^{n \times h}$分别计算如下：
$$
\begin{split}\begin{aligned}
\boldsymbol{I}_t &= \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xi} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hi} + \boldsymbol{b}_i),\\
\boldsymbol{F}_t &= \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xf} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hf} + \boldsymbol{b}_f),\\
\boldsymbol{O}_t &= \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xo} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{ho} + \boldsymbol{b}_o),
\end{aligned}\end{split}
$$
其中的 $\boldsymbol{W}_{xi}, \boldsymbol{W}_{xf}, \boldsymbol{W}_{xo} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hi}, \boldsymbol{W}_{hf}, \boldsymbol{W}_{ho} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_i, \boldsymbol{b}_f, \boldsymbol{b}_o \in \mathbb{R}^{1 \times h}$是偏差参数。

## 1.2. 候选记忆细胞
接下来，长短期记忆需要计算候选记忆细胞$\tilde{\boldsymbol{C}}_t$。它的计算与上面介绍的3个门类似，但使用了值域在 $[−1,1]$ 的tanh函数作为激活函数，如图所示。

<img src="https://zh.gluon.ai/_images/lstm_1.svg" width="500"/>

<center> 长短期记忆中候选记忆细胞的计算</center>

具体来说，时间步 $t$ 的候选记忆细胞$\tilde{\boldsymbol{C}}_t \in \mathbb{R}^{n \times h}$的计算为
$$
\tilde{\boldsymbol{C}}_t = \text{tanh}(\boldsymbol{X}_t \boldsymbol{W}_{xc} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hc} + \boldsymbol{b}_c),
$$
其中$\boldsymbol{W}_{xc} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hc} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_c \in \mathbb{R}^{1 \times h}$是偏差参数。

##  1.3. 记忆细胞
我们可以通过元素值域在 $[0,1]$ 的输入门、遗忘门和输出门来控制隐藏状态中信息的流动，这一般也是通过使用按元素乘法（符号为$\odot$)来实现的。当前时间步记忆细胞 $\boldsymbol{C}_t \in \mathbb{R}^{n \times h}$的计算组合了上一时间步记忆细胞和当前时间步候选记忆细胞的信息，并通过遗忘门和输入门来控制信息的流动：
$$
\boldsymbol{C}_t = \boldsymbol{F}_t \odot \boldsymbol{C}_{t-1} + \boldsymbol{I}_t \odot \tilde{\boldsymbol{C}}_t.
$$
如图所示，遗忘门控制上一时间步的记忆细胞$\boldsymbol{C}_{t-1}$中的信息是否传递到当前时间步，而输入门则控制当前时间步的输入 $X_t$ 通过候选记忆细胞$\tilde{\boldsymbol{C}}_t$如何流入当前时间步的记忆细胞。如果遗忘门一直近似1且输入门一直近似0，过去的记忆细胞将一直通过时间保存并传递至当前时间步。这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。

<img src="https://zh.gluon.ai/_images/lstm_2.svg" width="500"/>

<center>长短期记忆中记忆细胞的计算。这里的 $\odot$是按元素乘法</center>

## 1.4. 隐藏状态
有了记忆细胞以后，接下来我们还可以通过输出门来控制从记忆细胞到隐藏状态$\boldsymbol{H}_t \in \mathbb{R}^{n \times h}$ 的信息的流动：
$$
\boldsymbol{H}_t = \boldsymbol{O}_t \odot \text{tanh}(\boldsymbol{C}_t).
$$
这里的tanh函数确保隐藏状态元素值在-1到1之间。需要注意的是，当输出门近似1时，记忆细胞信息将传递到隐藏状态供输出层使用；当输出门近似0时，记忆细胞信息只自己保留。下图展示了长短期记忆中隐藏状态的计算。

<img src="https://zh.gluon.ai/_images/lstm_3.svg" width="500"/>

<center> 长短期记忆中隐藏状态的计算。这里的 ⊙ 是按元素乘法</center>

# 2. 读取数据集
下面我们开始实现并展示长短期记忆。和前几节中的实验一样，这里依然使用周杰伦歌词数据集来训练模型作词。

```python
%matplotlib inline
import math
import tensorflow as tf
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import losses
from tensorflow.data import Dataset
import time
import random
import zipfile
```

```python
tf.enable_eager_execution() #tf2不用
```

```python
def load_data_jay_lyrics():
  from google.colab import drive
  drive.mount('/content/drive')
  with zipfile.ZipFile('/content/drive/My Drive/data/d2l-zh-tensoflow/jaychou_lyrics.txt.zip')as zin:
    with zin.open('jaychou_lyrics.txt') as f:
      corpus_chars=f.read().decode('utf-8')
  corpus_chars=corpus_chars.replace('\n',' ').replace('\r',' ')
  corpus_chars=corpus_chars[0:10000]
  idx_to_char=list(set(corpus_chars))
  char_to_idx=dict([(char,i) for i,char in enumerate(idx_to_char)])
  vocab_size=len(char_to_idx)
  corpus_indices=[char_to_idx[char] for char in corpus_chars]
  return corpus_indices,char_to_idx,idx_to_char,vocab_size

(corpus_indices,char_to_idx,idx_to_char,vocab_size)=load_data_jay_lyrics() 
```

##  3. 从零开始实现LSTM
我们先介绍如何从零开始实现长短期记忆。

## 3.1. 初始化模型参数
下面的代码对模型参数进行初始化。超参数num_hiddens定义了隐藏单元的个数。

```python
num_inputs,num_hiddens,num_outputs=vocab_size,256,vocab_size
def get_params():
  def _one(shape):
    return tf.Variable(tf.random.normal(stddev=0.01,shape=shape))

  def _three():
    return (_one((num_inputs,num_hiddens)),
        _one((num_hiddens,num_hiddens)),
        tf.Variable(tf.zeros(num_hiddens)))
    
  W_xi,W_hi,b_i=_three() #输入门参数
  W_xf,W_hf,b_f=_three() #遗忘门参数
  W_xo,W_ho,b_o=_three() #输出门参数
  W_xc,W_hc,b_c=_three() #候选记忆细胞参数

  #输出层参数
  W_hq=_one((num_hiddens,num_outputs))
  b_q=tf.Variable(tf.zeros(num_outputs))

  params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
  return params
```



# 4. 定义模型
在初始化函数中，长短期记忆的隐藏状态需要返回额外的形状为(批量大小, 隐藏单元个数)的值为0的记忆细胞。

```python
def init_lstm_state(batch_size,num_hiddens):
  return (tf.zeros(shape=(batch_size,num_hiddens)),tf.zeros(shape=(batch_size,num_hiddens)))
```

下面根据长短期记忆的计算表达式定义模型。需要注意的是，只有隐藏状态会传递到输出层，而记忆细胞不参与输出层的计算。

```python
def lstm(inputs,state,params):
  W_xi,W_hi,b_i,W_xf,W_hf,b_f,W_xo,W_ho,b_o,W_xc,W_hc,b_c,W_hq,b_q=params
  (H,C)=state
  outputs=[]

  for X in inputs:
    I=tf.sigmoid(tf.matmul(X,W_xi)+tf.matmul(H,W_hi)+b_i)
    F=tf.sigmoid(tf.matmul(X,W_xf)+tf.matmul(H,W_hf)+b_f)
    O=tf.sigmoid(tf.matmul(X,W_xo)+tf.matmul(H,W_ho)+b_o)
    C_tilda=tf.tanh(tf.matmul(X,W_xc)+tf.matmul(H,W_hc)+b_c)
    C=F*C+I*C_tilda
    H=O*tf.tanh(C)
    Y=tf.matmul(H,W_hq)+b_q
    outputs.append(Y)
  return outputs,(H,C)
```

```python
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']
```

## 4.1. 训练模型并创作歌词
同上一节一样，我们在训练模型时只使用相邻采样。设置好超参数后，我们将训练模型并根据前缀“分开”和“不分开”分别创作长度为50个字符的一段歌词。

```python
def to_onehot(X,size):
  return [tf.one_hot(x,size) for x in tf.transpose(X)]
```

```python
def predict_rnn(prefix,num_chars,rnn,params,init_rnn_state,num_hiddens,vocab_size,idx_to_char,char_to_idx):
  state=init_rnn_state(1,num_hiddens)
  output=[char_to_idx[prefix[0]]]
  for t in range(num_chars+len(prefix)-1):
    #将上一时间步的输出作为当前时间步的输入
    X=to_onehot(tf.reshape(tf.constant([output[-1]]),shape=(1,1)),vocab_size)
    # print(X[0].shape)
    #计算输出和更新隐藏状态
    (Y,state)=rnn(X,state,params)
    #下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
    if t<len(prefix)-1:
      output.append(char_to_idx[prefix[t+1]])
    else:
      output.append(tf.argmax(Y[0],axis=1).numpy()[0])
  return ''.join([idx_to_char[i] for i in output])
```

```python
def data_iter_random(corpus_indices,batch_size,num_steps):
  #减1是因为输出的索引是相应输入的索引加1
  num_examples=(len(corpus_indices)-1)//num_steps
  epoch_size=num_examples//batch_size
  example_indices=list(range(num_examples))
  random.shuffle(example_indices)

  #返回从pos开始的长为num_steps的序列
  def _data(pos):
    return corpus_indices[pos:pos+num_steps]

  for i in range(epoch_size):
    #每次读取batch_size个随机样本
    i=i*batch_size
    batch_indices=example_indices[i:i+batch_size]
    X=[_data(j*num_steps) for j in batch_indices]
    Y=[_data(j*num_steps+1) for j in batch_indices]
    yield tf.constant(X),tf.constant(Y)

def data_iter_consecutive(corpus_indices,batch_size,num_steps):
  corpus_indices=tf.constant(corpus_indices)
  data_len=len(corpus_indices)
  batch_len=data_len//batch_size
  indices=tf.reshape(corpus_indices[0:batch_size*batch_len],shape=(batch_size,batch_len))
  epoch_size=(batch_len-1) // num_steps
  for i in range(epoch_size):
    i=i*num_steps
    X=indices[:,i:i+num_steps]
    Y=indices[:,i+1:i+num_steps+1]
    yield X,Y    
```

```python
def sgd(params,l,t,lr,batch_size,theta):
  norm=tf.constant([0],dtype=tf.float32)
  for param in params:
    dl_dp=t.gradient(l,param)
    if dl_dp is None:
      print(param,dl_dp)
    norm+=tf.reduce_sum((dl_dp**2))
  norm=tf.sqrt(norm).numpy()
  if norm>theta:
    for param in params:
      dl_dp=t.gradient(l,param) #求梯度
      dl_dp=tf.assign(tf.Variable(dl_dp),dl_dp*theta/norm) #裁剪梯度
      param.assign_sub(lr*dl_dp/batch_size) #更新梯度
```

```python
def train_and_predict_rnn(rnn,get_params,init_rnn_state,num_hiddens,vocab_size,corpus_indices,idx_to_char,char_to_idx,is_random_iter,num_epochs,
                          num_steps,lr,clipping_theta,batch_size,pred_period,pred_len,prefixes):
  if is_random_iter:
    data_iter_fn=data_iter_random
  else:
    data_iter_fn=data_iter_consecutive

  params=get_params()
  loss=losses.SparseCategoricalCrossentropy(from_logits=True)

  for epoch in range(num_epochs):
    if not is_random_iter:#如果使用相邻采样，在epoch开始时初始化隐藏状态
      state=init_rnn_state(batch_size,num_hiddens)
    l_sum,n,start=0.0,0,time.time()
    data_iter=data_iter_fn(corpus_indices,batch_size,num_steps)
    for X,Y in data_iter:
      if is_random_iter:#如果使用相邻采样，在每个小批量更新前初始化隐藏状态
        state=init_rnn_state(batch_size,num_hiddens)
      else:#否则需要使用detach函数从
        state=tf.stop_gradient(state) #停止计算该张量的梯度
      with tf.GradientTape(persistent=True) as t:
        t.watch(params)
        inputs=to_onehot(X,vocab_size)
        # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
        (outputs,state)=rnn(inputs,state,params)
        # 拼接之后形状为(num_steps * batch_size, vocab_size)
        outputs=tf.concat(values=[*outputs],axis=0)
        # Y的形状是(batch_size, num_steps)，转置后再变成长度为
        # batch * num_steps 的向量，这样跟输出的行一一对应
        y=tf.reshape(tf.transpose(Y),shape=(-1,))
        #使用交叉熵损失计算平均分类误差
        l=tf.reduce_mean(loss(y,outputs))
      sgd(params,l,t,lr,1,clipping_theta) #因为误差已经取过均值了,所以batch_size为1
      l_sum+=l.numpy()*y.numpy().shape[0]
      n+=y.numpy().shape[0]
    if(epoch +1)%10==0:
      print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, l_sum / n, time.time() - start))
      for prefix in prefixes:
          print(' -', predict_rnn(
              prefix, pred_len, rnn, params, init_rnn_state,
              num_hiddens, vocab_size, idx_to_char, char_to_idx))
```

```python
train_and_predict_rnn(lstm,get_params,init_lstm_state,num_hiddens,vocab_size,corpus_indices,idx_to_char,char_to_idx,False,num_epochs,num_steps,lr,clipping_theta,batch_size,pred_period,pred_len,prefixes)
```

![image-20200903161534217](http://qfth8dccq.hn-bkt.clouddn.com/images/image-20200903161534217.png)

# 5. 简洁实现

```python
BATCH_SIZE=64
BUFFER_SIZE=1000
seq_length=100
def make_dataset():
  examples_per_epoch=len(corpus_indices) //seq_length
  char_dataset=Dataset.from_tensor_slices(np.array(corpus_indices))
  sequences=char_dataset.batch(seq_length+1,drop_remainder=True)
  def split_input_target(chunk):
    input_text=chunk[:-1]
    target_text=chunk[1:]
    return input_text,target_text
  dataset=sequences.map(split_input_target)
  setps_per_epoch=examples_per_epoch // BATCH_SIZE
  dataset=dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)
  return dataset,setps_per_epoch
dataset,setps_per_epoch=make_dataset()
for x in dataset:
  print(x)
  break
```

```python
def train_model(num_hiddens=256,embedding_dim=256,epochs=500):
  net=keras.Sequential()
  net.add(keras.layers.Embedding(input_dim=vocab_size,output_dim=vocab_size,batch_input_shape=(BATCH_SIZE,seq_length)))
  net.add(keras.layers.LSTM(num_hiddens,unroll=True,return_sequences=True,stateful=True))
  net.add(keras.layers.Dense(vocab_size,activation='softmax'))
  net.compile(optimizer=keras.optimizers.Adam(),loss=losses.SparseCategoricalCrossentropy(),metrics=['acc'])
  net.fit_generator(dataset.repeat(),steps_per_epoch=setps_per_epoch,epochs=epochs)
  return net
```

```
net=train_model()
```

```python
def generate_text(source_net,pred_len=50,prefix='分开'):
  #因为训练的网络是stateful，在keras中使用其预测时输入的向量shape必须跟训练时输入的向量shape一致,
  #但是这里我们我们生成文本只需要输入几个前缀文字，因此重新定义一个新模型，并修改输向量的shape，然后使用原有模型的权重
  num_hiddens=256
  new_net=keras.Sequential()
  new_net.add(keras.layers.Embedding(input_dim=vocab_size,output_dim=vocab_size,batch_input_shape=(1,1)))
  new_net.add(keras.layers.LSTM(num_hiddens,unroll=True,return_sequences=True,stateful=True))
  new_net.add(keras.layers.Dense(vocab_size,activation='softmax'))
  new_net.set_weights(source_net.get_weights())
  new_net.compile(optimizer=keras.optimizers.Adam(),loss=losses.SparseCategoricalCrossentropy(),metrics=['acc'])
  text_generated=prefix
  for i in range(pred_len):
    id=char_to_idx[text_generated[-1]]
    char=idx_to_char[tf.argmax(new_net.predict(tf.constant(value=[id]))[0],axis=-1).numpy()[0]]
    text_generated+=char
  return text_generated
```

![image-20200903161734237](http://qfth8dccq.hn-bkt.clouddn.com/images/image-20200903161734237.png)

# 6. 小结

* 长短期记忆的隐藏层输出包括隐藏状态和记忆细胞。只有隐藏状态会传递到输出层。
* 长短期记忆的输入门、遗忘门和输出门可以控制信息的流动。
* 长短期记忆可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。



**代码：**https://github.com/chongzicbo/Dive-into-Deep-Learning-tf.keras/blob/master/6.8.%20%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86(LSTM).ipynb

# 参考

[1]《动手学深度学习》