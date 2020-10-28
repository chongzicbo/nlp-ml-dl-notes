<center><b><font color=#A52A2A size=5 >公众号：数据挖掘与机器学习笔记</font></b></center>

# 1.简介

ELMO是一种深层的上下文单词表示模型，它可以同时建模：

(1) 单词使用的复杂特征（例如语法和语义），也就是能够学习到词汇用法的复杂性

(2)这些用法如何在语言上下文之间变化（即建模多义性）

词向量是深度双向语言模型(deep bidirectional language model,BILM)内部状态的可学习函数，这些双向语言模型在大型文本语料库上进行了预训练。可以将这些预训练的词向量添加到现有模型中，能够显著改善NLP问题(问答、文本蕴含、情感分析等)的解决效果。

# 2.ELMO的显著特征

* 依赖于上下文(Contextual):每个单词的表示形式最终取决于使用该单词的整个上下文。
* 深(deep):单词的向量表示结合了深度预训练神经网络的所有层。
* 基于字符(Character based):ELMO模型的训练完全基于字符，因此网络可以使用形态学线索来为训练中未曾见过的词汇标记成可靠的表示。

# 3.双向语言模型

前向语言模型就是，已知![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88t_1%2Ct_2%2C...%2Ct_%7Bk-1%7D%29) ，预测下一个词语$t_k$  的概率，写成公式就是

![[公式]](https://www.zhihu.com/equation?tex=p%28t_1%2Ct_2%2C...%2Ct_N%29%3D%5Cprod_%7Bk%3D1%7D%5E%7BN%7Dp%28t_k%7Ct_1%2Ct_2%2C...%2Ct_%7Bk-1%7D%29.)

后向的语言模型如下，即通过下文预测之前的 词语：

![[公式]](https://www.zhihu.com/equation?tex=p%28t_1%2Ct_2%2C...%2Ct_N%29%3D%5Cprod_%7Bk%3D1%7D%5E%7BN%7Dp%28t_k%7Ct_%7Bk%2B1%7D%2Ct_%7Bk%2B2%7D%2C...%2Ct_%7BN%7D%29.)

双向语言模型（biLM）将前后向语言模型结合起来，最大化前向、后向模型的联合似然函数即可，如下式所示：

![img](https://pic4.zhimg.com/v2-e29066a7b8fc29f769ac65f126d2fb17_r.jpg)

向左和向右的LSTM是不同的, 也就是说有两个LSTM单元.
 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_x) 是输入的意思. 输入的内容是最初始的词向量.   ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_s)是输出内容, 即LSTM在每个位置的 h. h会再用作softmax的输入, 来进行词汇选择权重赋予.

# 4.ELMO

ELMO 的本质思想是：我事先用语言模型学好一个单词的 Word Embedding，此时多义词无法区分，不过这没关系。在我实际使用 Word Embedding 的时候，单词已经具备了特定的上下文了，这个时候我可以根据上下文单词的语义去调整单词的 Word Embedding 表示，这样经过调整后的 Word Embedding 更能表达在这个上下文中的具体含义，自然也就解决了多义词的问题了。所以 ELMO 本身是个根据当前上下文对 Word Embedding 动态调整的思路。

![image-20201023222456669](https://gitee.com/chengbo123/images/raw/master/image-20201023222456669.png)

ELMo是双向语言模型biLM的多层表示的组合，对于某一个词语$t_k$  ，一个$L$层的双向语言模型biLM能够由2$L$+1个向量表示：

![[公式]](https://www.zhihu.com/equation?tex=R_k+%3D+%5C%7BX%5E%7BLM%7D%2C%5Coverrightarrow%7Bh%7D%5E%7BLMj%7D_%7Bk%7D%2C+%5Coverleftarrow%7Bh%7D%5E%7BLMj%7D_%7Bk%7D%7Cj%3D1%2C...%2CL%5C%7D%3D%5C%7B%7Bh%7D%5E%7BLMj%7D_%7Bk%7D%2C+%7Cj%3D1%2C...%2CL%5C%7D)

其中，![[公式]](https://www.zhihu.com/equation?tex=x_k%5E%7BLM%7D)是对token进行直接编码的结果(这里是字符通过CNN编码)，![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbf%7Bh%7D_%7Bk%2C+0%7D%5E%7BL+M%7D)代表$x_k^{LM}$，![[公式]](https://www.zhihu.com/equation?tex=h_%7Bk%2Cj%7D%5E%7BLM%7D+%3D+%5B%5Coverrightarrow%7Bh%7D_%7Bk%2Cj%7D%5E%7BLM%7D%3B+%5Coverleftarrow%7Bh%7D_%7Bk%2C+j%7D%5E%7BLM%7D%5D)是每个biLSTM层输出的结果。最上面一层的输出 ![[公式]](https://www.zhihu.com/equation?tex=%5Coverrightarrow%7B%5Cmathbf%7Bh%7D%7D_%7Bk%2C+L%7D%5E%7BL+M%7D)是用softmax来预测下面一个单词$t_{k+1}$。

最后对每一层LSTM的向量进行线性组合，得到最终的向量：

![img](https://pic3.zhimg.com/v2-e52a057fc33a0284b0b13be7f71958a6_r.jpg)

其中$s^{task}$是一个softmax出来的结果, γ是一个任务相关的scale参数。

# 5. ELMO训练好的向量如何使用

ELMO 采用了典型的两阶段过程，第一个阶段是利用语言模型进行预训练；第二个阶段是在做下游任务时，从预训练网络中提取对应单词的网络各层的 Word Embedding 作为新特征补充到下游任务中。

使用上述网络结构利用大量语料做语言模型任务就能预先训练好这个网络，如果训练好这个网络后，输入一个新句子Snew，句子中每个单词都能得到对应的**三个Embedding**:最底层是单词的 Word Embedding，往上走是第一层双向LSTM中对应单词位置的 Embedding，这层编码单词的**句法信息**更多一些；再往上走是第二层LSTM中对应单词位置的 Embedding，这层编码单词的**语义信息**更多一些。也就是说，ELMO 的预训练过程不仅仅学会单词的 Word Embedding，还学会了一个双层双向的LSTM网络结构，而这两者后面都有用。对于这三个EMbedding向量，可以对其进行加权组合成一个向量，然后作为下游任务的输入。因为 ELMO给下游提供的是每个单词的特征形式，所以这一类预训练的方法被称为“Feature-based Pre-Training”。

![image-20201023225633213](https://gitee.com/chengbo123/images/raw/master/image-20201023225633213.png)

# 6.ELMO的缺点

* 在特征抽取器选择上，LSTM弱于Transformer
* ELMO 采取双向拼接这种融合特征的能力可能比 Bert 一体化的融合特征方式弱



参考：

[1]https://allennlp.org/elmo

[2]https://zhuanlan.zhihu.com/p/37684922

[3]https://zhuanlan.zhihu.com/p/38254332

[4]https://zhuanlan.zhihu.com/p/63115885



![在这里插入图片描述](https://img-blog.csdnimg.cn/20200828221113544.jpg#pic_center)

