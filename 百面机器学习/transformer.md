## 写在前面 

前些时间，赶完论文，开始对 Transformer、GPT、Bert 系列论文来进行仔仔细细的研读，然后顺手把站内的相关问题整理了一下，但是发现站内鲜有回答仔细的~所以自己就在网上针对每个问题收集了一些资料，并做了整理，有些问题还写了一些自己的看法，可能会有纰漏，甚至还有错误，还请大家赐教 😊

模型总览：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/DHibuUfpZvQdyJ7ZK5OKBf0byZZp7icU8xvGZx24EwQM60e0HhEQxb1RYJMRyuwicSvwIV1dVPKDxddH8sJ7GZPvw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)Transformer模型总览

## 1.Transformer 的结构是什么样的？

Transformer 本身还是一个典型的 encoder-decoder 模型，如果从模型层面来看，Transformer 实际上就像一个 seq2seq with attention 的模型，下面大概说明一下 Transformer 的结构以及各个模块的组成。

#### 1.1 Encoder 端 & Decoder 端总览



- Encoder 端由 N(原论文中「N=6」)个相同的大模块堆叠而成，其中每个大模块又由「两个子模块」构成，这两个子模块分别为多头 self-attention 模块，以及一个前馈神经网络模块；

- - 「需要注意的是，Encoder 端每个大模块接收的输入是不一样的，第一个大模块(最底下的那个)接收的输入是输入序列的 embedding(embedding 可以通过 word2vec 预训练得来)，其余大模块接收的是其前一个大模块的输出，最后一个模块的输出作为整个 Encoder 端的输出。」



- Decoder 端同样由 N(原论文中「N=6」)个相同的大模块堆叠而成，其中每个大模块则由「三个子模块」构成，这三个子模块分别为多头 self-attention 模块，「多头 Encoder-Decoder attention 交互模块」，以及一个前馈神经网络模块；

- - 同样需要注意的是，Decoder端每个大模块接收的输入也是不一样的，其中第一个大模块(最底下的那个)训练时和测试时的接收的输入是不一样的，并且每次训练时接收的输入也可能是不一样的(也就是模型总览图示中的"shifted right"，后续会解释)，其余大模块接收的是同样是其前一个大模块的输出，最后一个模块的输出作为整个Decoder端的输出

  - 对于第一个大模块，简而言之，其训练及测试时接收的输入为：

  - - 训练的时候每次的输入为上次的输入加上输入序列向后移一位的 ground truth(例如每向后移一位就是一个新的单词，那么则加上其对应的 embedding)，特别地，当 decoder 的 time step 为 1 时(也就是第一次接收输入)，其输入为一个特殊的 token，可能是目标序列开始的 token(如)，也可能是源序列结尾的 token(如)，也可能是其它视任务而定的输入等等，不同源码中可能有微小的差异，其目标则是预测下一个位置的单词(token)是什么，对应到 time step 为 1 时，则是预测目标序列的第一个单词(token)是什么，以此类推；

    - - 这里需要注意的是，在实际实现中可能不会这样每次动态的输入，而是一次性把目标序列的embedding通通输入第一个大模块中，然后在多头attention模块对序列进行mask即可

- 

- - - 而在测试的时候，是先生成第一个位置的输出，然后有了这个之后，第二次预测时，再将其加入输入序列，以此类推直至预测结束。

#### 1.2 Encoder 端各个子模块

「1.2.1 多头 self-attention 模块」

在介绍 self-attention 模块之前，先介绍 self-attention 模块，图示如下：

![img](https://mmbiz.qpic.cn/mmbiz_jpg/DHibuUfpZvQdyJ7ZK5OKBf0byZZp7icU8xk4Yz5naOHL7Gwl7ZRayzlMrKGBHiaTqcNzW4ssGQxpyWBjKSITtoiaGg/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)self-attention

上述 attention 可以被描述为「将 query 和 key-value 键值对的一组集合映射到输出」，其中 query，keys，values 和输出都是向量，其中 query 和 keys 的维度均为$d_k$,values 的维度为$d_v$(论文中$d_k=d_v=d_{model}/h=64$)，输出被计算为 values 的加权和，其中分配给每个 value 的权重由 query 与对应 key 的相似性函数计算得来。这种 attention 的形式被称为“Scaled Dot-Product Attention”，对应到公式的形式为：
$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt {d_k}})V
$$




来源：https://mp.weixin.qq.com/s/x9ZCIQH78XlIBmIBZqwXcw



