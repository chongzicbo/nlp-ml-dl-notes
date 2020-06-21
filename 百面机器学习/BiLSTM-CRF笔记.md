## 1.基本介绍

### 1.1 标签简介

假设有一个数据集，其中有两个实体类型，Person和Organization。事实上在数据集中，有5个实体标签：

* B-Person

* I-Person

* B-Organization

* I-Organization

* O 
x是一个包含5个单词的句子, $w_0,w_1,w_2,w_3,w_4$,在句子x中，$[w_0，w_1]$是一个Person实体，$[w_3]$是一个Organization实体，其他都是“O”

### 1.2 BiLSTM-CRF模型

输入：单词向量，包括单词嵌入和字符嵌入。字符嵌入是随机初始化，词嵌入是预先训练好的。所有嵌入在训练过程中进行微调。
输出：x中单词的预测标签



![](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvogVFNllnEibbMIA3SW8nO2Sb7iagAo9icRBvHHgxNzeGnMdMUUlxb8veq60aMchV5kOwaFcA5aT4ibFA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![](https://mmbiz.qpic.cn/mmbiz_png/KYSDTmOVZvogVFNllnEibbMIA3SW8nO2SrrZgpyjpxzyrlEibaUPFcYc8CjGgQd8UtWiaUNrNF4tHqr3iagFw3FxOw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
输入是词向量，输出是每个标签的分数。然后将BiLSTM预测的分数输入CRF层。在CRF层中预测得分最高的标签序列作为最佳答案。

### 1.3 如果没有CRF层

![](https://mmbiz.qpic.cn/mmbiz_jpg/KYSDTmOVZvogVFNllnEibbMIA3SW8nO2SoyZFC2kxpMoPjRKBQNYxxbb1a5Ns4ufxIqRPyBiahN3Eg84SlzoqKdQ/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
没有CRF层也可以，BiLSTM输出每个单词每个标签的分数，可以选择得分最高的标签作为最终的预测标签。此时BiLSTM层的最后一般接的Softmax。但是这样会产生错误的标签顺序，比如“I-Organization I-Person”和“B-Organization I-Person”。



### 1.4 CRF层可以从训练数据中学到标签约束

不加CRF层，直接使用softmax会产生无效的标签顺序。CRF层可以在预测的标签之间添加约束，确保标签有效，这些约束可以右CRF层在训练过程中自动学习得到。

约束条件可以是：

- 句子中第一个单词的标签应该以“B-”或“O”开头，而不是“I-”
- “B-label1 I-label2 I-label3 I-…”，在这个模式中，label1、label2、label3…应该是相同的命名实体标签。例如，“B-Person I-Person”是有效的，但是“B-Person I-Organization”是无效的。
- “O I-label”无效。一个命名实体的第一个标签应该以“B-”而不是“I-”开头，换句话说，有效的模式应该是“O B-label”

 ## 2. CRF层

### 2.1 Emission得分

emission分数来自BiLSTM层，即BiLSTM层输出的各个单词的标签得分。

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9LWVNEVG1PVlp2cDRpY2NOb2lhUmRSWFd1WTFYUUpmYlhpYnZ2ZWliRlpUUGJyRXBpYnh5enFOc1dpYTZjQ3VuVzgzdEt3UGtNdUFKTTBpY25QTWliU2lheEFxRndaUS82NDA?x-oss-process=image/format,png)
为方便起见，给每个标签一个索引号，如：

![](https://pic2.zhimg.com/v2-d6ed1cadb710ae868f4582722dc7810d_r.jpg)
$x_{iy_j}$表示emission分数。i是单词索引，$y_j$是label索引，即单词i为标签$y_j$的得分

### 2.2 Transition得分

使用$y_{y_iy_j}$表示transition得分，表示从标签$y_i$转移到标签$y_j$的得分。有一个矩阵，存储了所有标签之间的转移得分情况。为了使transition评分矩阵更健壮，我们将添加另外两个标签，START和END。START是指一个句子的开头，而不是第一个单词。END表示句子的结尾。

下面是一个transition得分矩阵的例子，包括额外添加的START和END标签。

![](https://pic3.zhimg.com/v2-b8e713dbf24afb0431f5670e10c94a86_r.jpg)
通过transition矩阵，可以学习到一些有用约束。transition矩阵是在模型训练过程中学习而来，输入BiLSTM-CRF模型的一个参数。在训练之前，随机初始化transition矩阵，训练过程中该矩阵会自动更新，最终趋于合理。

### 2.3 CRF损失函数

CRF损失函数由真实路径得分和所有可能路径的总得分组成，在所有可能的路径中，真实路径的得分最高。

假设每条可能的路径都有一个分数$P_i$，并且总共有N条可能的路径，所有路径的总得分是
$$
P_{total}=P_1+P_2+\dots+P_N=e^{S_1}+e^{S_2}+\ldots+e^{S_N}
$$
如果说第N条路径是真正的路径，那在所有可能的路径中，得分$P_{10}$应该是百分比最大的。

在训练过程中，BiLSTM-CRF模型的参数值将会不断更新，以保持增加真实路径的分数百分比。
$$
LossFunction=\frac{P_{RealPath}}{P_1+P_2+\dots+P_N}
$$
那么(1)如何定义一个路径的分数？（2）如何计算所有可能的路径总分？（3）当计算总分时，需要列出所有可能的路径吗？(这个问题是否定的)

### 2.4 实际路径得分

在所有可能的N条路径中，显然只有一条是真实路径。

在训练过程中，CRF损失函数只需要两个分数：真实路径分数和所有可能路径的总分数。真实路径得分占比在训练过程中会不断增大。那么$S_i$该怎么计算呢？

![](https://pic4.zhimg.com/v2-e3255536f84eaf2e5cf295de0ad2c17b_r.jpg)

### 2.5 所有可能路径的得分

最简单的方法：列举所有可能的路径并将分数相加。时间复杂度很高。

把loss函数变成log loss函数，目标是最小化损失函数，因此加上一个负号：
$$
log LossFunction=-log\frac{P_{RealPath}}{P_1+P_2+\ldots+P_N}=-log\frac{e^{S_{RealPath}}}{e^{S_1}+e^{S_2}+\dots+e^{S_N}}
$$
现在，目标是计算
$$
log(e^{S_1}+e^{S_2}+\dots+e^{S_N})
$$
![](https://pic4.zhimg.com/v2-cd4b642818feed74de3410daeecedc6b_r.jpg)

![](https://pic1.zhimg.com/v2-3441d09333a449e8ecaf620a1ea574a4_r.jpg)

![](https://pic1.zhimg.com/v2-2807822cc1293b2bfd8e5a107ae0a088_r.jpg)

![](https://pic3.zhimg.com/v2-41bd4f0249925ce774d65f4e03d5d4f6_r.jpg)

![](https://pic3.zhimg.com/v2-f7233db17ebbb5a4d904bf261cea89ca_r.jpg)

![](https://pic4.zhimg.com/v2-8d7a12744e48116451a1de50ffd87bb7_r.jpg)

![](https://pic4.zhimg.com/v2-3baea8e873dcfe6dcfc8ed5d7b689f53_r.jpg)

![](https://pic1.zhimg.com/v2-95260da313998d9121894e07083321d4_r.jpg)