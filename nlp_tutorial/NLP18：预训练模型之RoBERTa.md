# 1.摘要

预训练模型让nlp任务的效果得到了极大提升，但不同方法之间的比较也很困难。预训练模型的训练通常在不同大小的私有数据集上完成，计算代价很昂贵，超参数的选择对最终的效果具有显著影响。我们提出了一个BERT的复制研究，精心评估衡量大量关键超参数和训练数据大小的影响。我们发现BERT明显训练不足，它的效果能够达到或者超过在它之后发布的每个模型的性能。我们训练出的最好的模型在GLUE、RACE和SQuAD上能达到SOTA。模型结果突出了之前被忽视的设计选择的重要性，并提出了模型为什么能够改进的问题。

# 2. 引言

ELMO、GPT、XLM、XLNet等自训练方法取得了极大的性能提升，但在如何确定那种方法贡献最大上还存在困难。模型的训练在计算上很昂贵，限制了调参的数量。训练数据通常是私有的且大小不一致，导致无法评估模型的效果。

在BERT的基础上提出了Roberta模型，对不同的超参数微调和训练集大小做了一个详细的评估。我们发现BERT寻找严重的训练不足，我们提出了新的训练方法，roberta模型可以达到或者超过BERT之后提出的方法的性能。Roberta模型做出的改变包括：

(1)使用更长的训练时间、更大的batch和更多的训练数据

(2)去除NSP训练目标

(3)在更长的sentence上进行训练

(4)训练数据上动态改变masking方式

收集了一个大的新的训练数据集(CC-News)。相比其它的私有训练数据集，可以更好地控制训练集大小地影响。

当控制训练数据时，我们改进后的训练程序改善了之前模型在GLUE和SQuAD上的结果。当在额外数据上训练更长的时间，我们的模型在公共GLUE榜单上取得了88.5的分数，与Yang et al的88.4分相匹配。我们的模型4/9的GLUE任务：MNLI,QNLI、RTE和STS-B上取得了SOTA。在SQuAD和RACE任务上也取得了SOTA。总而言之，模型重新构建了BERT的masked语言模型训练目标，与其他的训练目标如扰动自回归语言建模(perturbed autoregressive langage modeling)相比具有竞争力

总结Roberta主要有以下几点贡献：

(1)展示了一些重要的BERT设计选择、训练策略，并介绍了可以提升下游任务性能的选择。

(2)使用了新的数据集CC-NEWS,使用更多的数据用于预训练以进一步改善在下游任务上的性能。

(3)训练的改进表明在正确的设计选择下，masked语言模型预训练与最近发布的其它方法相比具有竞争力。

我们发布了PyTorch的Roberta预训练模型和微调代码。



# 3.Background

## 3.1 Setup

BERT将两个片段(token序列)：$x_1,\ldots,x_N$和$y_1,\ldots,y_M$拼接起来作为输入。片段通常由超过一个自然句子组成。将两个片段拼接成一个语句并使用特殊字符分隔后输入到BERT：
$$
[CLS],x_1,\ldots,x_N,[SEP],y_1,\ldots,y_M,[EOS].
$$
$M+N < T$，$T$用于控制训练期间的最长句长。

模型首先在一个大的非标注文本语料上进行预训练，随后在标注数据上进行微调任务。



## 3.2 Architecture

BERT使用transformer架构，RoBERTa使用L层的transformer架构。每个模块(block)使用A个自注意力头(self-attention heads),hidden维度为$H$

## 3.3 训练目标

BERT的预训练使用了两个目标：masked language modeling（MLM）和下一句预测(next sentence prediction,NSP)

* Masked Language Model(MLM)

  在输入句子的字符中随机选择一些并替换成“[MASK]”字符.MLM目标是一个交叉熵损失。BERT选择输入字符的15%进行替换，其中的80%被替换成"[MASK]"，10%不做改变，10%被随机替换成其它字符。

  在原始的实现中，随机Masking和替换仅在开始执行一次，然后在训练期间就固定下来。尽管数据重复，但是对于每个训练语句mask并不总是相同的。

* Next Sentence Prediction(NSP)

  NSP是一个二分类损失，用于预测是否两个片段在原始文本中连续。正样本在文本语料中选择连续的语句生成，负样本从不同文档中选择片段对。正负样本使用相同的概率采样。

  NSP目标被设计用于改善下游任务性能，如自然语言推理，其需要推理句子对之间的关系。

  ## 3.4 优化器

  BERT使用Adam优化器，参数如下：$\beta _1=0.9,\beta_2=0.999,\epsilon=1e-6$和$L_2$的权重衰减系数为0.01。使用warm up学习率，前10000步的学习率达到了$1e-4$的顶峰值，然后线性衰减。所有层、attention权重和GELU激活函数的dropout rate为0.1，模型预训练进行S=1000000次更新，mini-batches 包含B=256个句子，最大长度$T=512$个字符。

  ## 3.5 数据

  BERT的训练数据包含BOOKCORPUS和英文维基百科，总共16GB非压缩文本。

  

# 4.实验设置

描述BERT的复制版本RoBERTa的实现设置。

## 4.1 实现(implement)

&emsp;&emsp;在FAIRSEQ中实现了BERT。除了峰值学习率和warmup的步数根据每个设置分开设置，其它的主要安照BERT的参数进行。我们还发现训练对Aadm的epsilon值非常敏感，在一些样例(cases)中，在调整该值后获得了更好的性能和稳定性。同样，我们发现，当使用更大batch size训练时，设置$\beta_2=0.98$可以改善稳定性。

使用最大句长为512,训练时不随机插入短句，在前90%的更新中不减短句长。只使用全长(full-length)句子训练。

在DGX-1机器上使用混合精度浮点算法训练，每台机器有$8 \times 32GB$Nvidia V100 GPU。



## 4.2 数据

BERT风格的预训练依赖大量的文本。Baevski et al. (2019)认为增加训练数据可以改善端任务性能。也有一些工作(Radford et al., 2019;
Yang et al., 2019; Zellers et al., 2019)尝试使用更大和更多样的训练数据进行训练。那些额外使用的数据集并没有都被发布出来。对于我们的研究，我们关注于获取尽可能多的实验数据，以便我们能够更好地为每个比较匹配到综合质量和数量的数据。

&emsp;&emsp;考虑5个英文语料，总共160GB非压缩文本：

* BOOKCORPUS加上英文维基百科。
* CC-NEWS,从CommonCrawl News数据集中收集而来。该数据包含在2016年9月-2019年2月之间爬取的6300万英文新闻文章。过滤后76GB。
* OPEN WEB TEXT。从Reddit上至少有三个赞成的URL中提取的web内容。38GB。
* STORIES,31GB。

##  4.3 评估

跟之前的工作一样，使用下列三个benchmarks来评估模型在下游任务的性能。

* GLUE

* SQuAD

* RACE

  

  

# 5.训练过程分析

探索哪种选择对于成功预训练模型是重要的。一开始训练训练BERT模型与$BERT_{base}$使用同样的配置:$L=12,H=768,A=12,110M$参数。

## 5.1 静态和动态Masking

BERT依赖随机的masking来预测字符。原始的BERT在数据预处理期间执行一次masking，结果就是静态的mask。为例避免在每个epoch每个训练实例使用相同的mask,训练数据被重复10次，使得在40个epoch的训练期间，每个字符序列以10种不同的方式进行mask。因此，在训练期间，每个训练字符序列会被重复使用4次。

我们将这种策略与动态masking进行比较，在动态masking中，字符序列在被输入进模型时会生成masking模式。当预训练步骤很多、数据集很大时，动态masking会更加重要。

![image-20210112123256110](https://gitee.com/chengbo123/images/raw/master/image-20210112123256110.png)



表1比较了$BERT_{base}$和我们实现的静态masking和动态masking。结果表明静态masking的结果跟$BERT_{base}$相当，而动态masking的结果相对较好。在随后的实验中均使用动态masking。

## 5.2 模型输入格式和下一句预测(NSP)

在原始的BERT程序中，模型的输入是两个拼接起来的片段，片段以0.5的概率相同和不同的文档中连续采样而来。除了MLM训练目标，模型还预测两个片段是否来自同一个文档，使用NSP损失。

在原始的BERT模型中，NSP损失是训练中的一个重要因素，Devlin et al. (2019) 观察到移除NSP损失有损性能，在QNLI, MNLI, and SQuAD 1.1上会有显著的性能下降。然而，最近的一些研究质疑了NSP损失的必要性(Lample and Conneau,
2019; Yang et al., 2019; Joshi et al., 2019)。

为了更好的理解这一差异，我们比较了集中训练格式：

* SEGMENT-PAIR+NSP

  BERT NSP损失的原始输入格式。每个输入是一对文本片段，每个片段包含多个自然语句，总的片段长度不超过512个字符。

* SENTENCE-PAIR+NSP

  每个输入包含一对自然语句，每个语句从一个文档采样连续的部分，或者从不同的文档采样。因为这样采样后的输入长度远小于512个字符，我们增大了batch size以便总的字符数与$SEGMENT-PAIR+NSP$保持相同。任然保留NSP损失。

* FULL-SENTENCES

  每个输入使用全句(full sentence) packed 而成，全句从一个或者更多的文档连续采样而成，因此总的长度最多512.输入可能跨文档边界。当到达一个文档末尾时，就从下一个文档开始采样sentences，然后在文档中间添加额外的分隔符。去除NSP损失。

* DOC-SENTENCES

  除了不跨文档边界，输入的构建类似于FULL-SENTENCES。从一个文档中采样直到文档末尾，由于可能少于512个字符，所以动态的增加batch size以便与FULL-SENTENCES具有相似数量的字符。移除NSP损失。

![image-20210112125712237](https://gitee.com/chengbo123/images/raw/master/image-20210112125712237.png)

SEGMENT-PAIR和SENTENCE-PAIR均保留NSP，后者使用单句，其性能在下游任务上要比原始BERT差，可能的原因是模型没能够学到长距离依赖。在移除NSP损失上，DOC-SENTENCES的效果要优于原始$BERT_{BASE}$,能够改善下游任务性能。可能的原因是原始的BERT保留SEGMENT-PAIR。此外，DOC-SENTENCES也比FULL-SENTENCE的效果稍好一点。然后因为DOC-SENTENCES使用不同的batch size，在后续的实验中使用FULL-SENTENCES方便比较。

## 5.3 使用更大的batches训练

之前神经机器翻译的工作显示，当学习率适当增大的话，使用大的mini-batches训练能够提高优化速度和端任务性能。最近的工作也表明BERT适用于大batch训练。

原始$BERT_{base}$使用batch_size=256训练了1M个steps。这在计算上等价于使用batch size=2K训练125K个steps，使用batch size=8K训练31K个steps。

![image-20210112162322044](https://gitee.com/chengbo123/images/raw/master/image-20210112162322044.png)



表3比较了$BERT_{base}$不同batch size和steps的困惑度和端任务性能。使用更大的batches可以改善MLM目标的困惑度和端任务准确率。大的batches也更容易并行训练。随后的训练使用batch size=8K。

## 5.4 文本编码

Byte-Pair Encoding(BPE)是介于字符级和词级之间的编码方式，可以让我们更方便地处理大的自然语言语料中的词汇。不同于full words，BPE依赖子词单元，子词单元通过对训练语料执行统计分析而来。

BPE词汇表的子词单元数量在10K-100K之间，然而，在建模大的和多样的语料时，unicode字符占了词汇表的很大一部分。Radford（2019）引入了一个BPE实现，使用字节（bytes）作为子词单元而不是unicode字符。使用字节使得词汇表大小在一个合理的数量(50K 单元),可以编码任何文本而不会产生未知的"字符"。

原始BERT实现了字符级的BPE，词汇表大小30K。我们按照Radford（2019）的做法，词汇表大小为50K个子词单元。对于$BERT_{base}$和$BERT_{LARGE}$这可能增加15M和20M参数。

# 6.RoBERTa

RoBERTa的训练方式如下：动态masking，FULL-SENTENCES去除NSP，更大的mini-batches和更大的byte-level BPE。除此之外，还考虑了两外两个被之前研究轻视的重要因素：

(1)预训练数据

(2)训练的steps数量

按照$BERT_{LARGE}$结构$L=24,H=1024,A=16,355M$参数训练RoBERTa，在BOOKCORPUS和WIKIPEDIA数据上训练100K个steps。预训练使用1024个V100GPU训练了将近一天。结果如下表：

![image-20210112165002463](https://gitee.com/chengbo123/images/raw/master/image-20210112165002463.png)

## 6.1 GLUE结果

对于GLUE，考虑两种finetuning设置，第一个设置(单任务,dev)，为每个GLUE任务单独finetune RoBERTa，相应任务仅使用训练数据。每个任务考虑超参数扫描。batch size为{16,32},学习率在{$1e-5,2e-5,3e-5$},学习率在前6%的steps里进行线性linear warmup，随后线性衰减到0。funetune10个epochs并启用early stopping。其余的参数在预训练期间保持不变。对于该设置，为每个人物在开发集结果上计算中位数。

第二个设置(集成设置,test)，将RoBERTa与其它的方法在测试集上通过CLUE leaderboard进行比较.

![image-20210112171818914](https://gitee.com/chengbo123/images/raw/master/image-20210112171818914.png)

## 6.2 SQuAD结果

![image-20210112172731849](https://gitee.com/chengbo123/images/raw/master/image-20210112172731849.png)

## 6.3 RACE结果

![image-20210112172806165](https://gitee.com/chengbo123/images/raw/master/image-20210112172806165.png)