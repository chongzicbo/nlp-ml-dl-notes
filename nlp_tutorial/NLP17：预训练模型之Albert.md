# 1. 摘要

在预训练自然语言表示时，增加模型大小通常会导致下游任务的性能提高。但是，由于 GPU/TPU 内存限制和更长的训练时间，在某些时候，进一步增大模型变得更加困难。为了解决这个问题，我们提出了两种参数削减技术来降低内存消耗以及加快模型训练速度。实验结果表明我们提出的方法比原始BERT模型的扩展性更好。使用自监督损失来建模句子间的连贯性，有助于下游任务的处理。在GLUE、RACE和SQuAD等benchmarks上取得了SOTA，而且与BERT-large相比拥有更少的参数。代码见：https://github.com/google-research/ALBERT.

# 2. 引言

深度学习神经网络预训练在语言表示学习方面取得了一系列突破。大量重要的NLP任务包括那些仅使用有限训练数据的任务都从预训练模型中受益匪浅。一个最大的突破就是在为中国初高中英语考试设计的阅读理解任务中机器性能的变化，the RACE test (Lai et al., 2017):最初的SOAT机器准确率为44.1%;目前最新的性能达到了 83.2% (Liu et al., 2019);而我们的工作达到了89.4，取得了45.3%的提升。主要的原因就是得益于高性能的预训练模型。

这些改进表明，大型网络对于实现高效的性能至关重要e (Devlin et al., 2019; Radford et al., 2019)。预训练模型现在已经变得越来越普遍，并通过蒸馏使他们变得更小从而适用于实际应用。鉴于模型尺寸的重要性，我们问：拥有更好的 NLP 模型是否像拥有更大的模型一样简单？（Is having better NLP models as easy as having larger models?）

回答这个问题的障碍是可用硬件的内存限制。鉴于当前最先进的模型通常具有数亿甚至数十亿个参数，因此在我们尝试扩展模型时，很容易达到这些限制。在分布式训练中，训练速度也会受到很大阻碍，因为通信开销与模型中的参数数成正比。

解决上述问题的现有办法包括模型并行化和更巧妙的内存管理。这些解决方案解决了内存限制问题，但解决不了通信开销问题。在这篇论文中，我们通过设计比传统 BERT 体系结构具有明显较少参数的精简 BERT （ALBERT） 体系结构来解决上述所有问题。

ALBERT 集成了两种参数缩减技术，可消除预训练模型精简的主要障碍。第一个是**嵌入参数分解**。通过将大型词汇嵌入矩阵分解为两个小矩阵，我们将隐藏层的大小与词汇嵌入的大小分开。这种分离使得在不显著增加词汇嵌入参数大小的同时增加隐藏大小更容易。第二个技术是**跨层参数共享**。此技术可防止参数随着网络深度而增长。这两种技术都显著减少了 BERT 的参数数量，而不会严重降低性能，从而提高参数效率。类似于 BERT-large 的 ALBERT 配置的参数减少了 18 倍，训练速度可提高约 1.7 倍。参数缩减技术还作为一种规范化形式，可稳定训练并有助于泛化。

为了进一步提高ALBERT的性能，我们还引入了句子顺序预测 （sentence-order prediction，SOP） 的自监督损失。SOP主要侧重于句间连贯性，旨在解决原始BERT中下一句预测（ next sentence prediction，NSP）的无效问题（。

使用上述设计，我们能够扩展至更大的 ALBERT 配置，这些配置的参数仍然比 BERT 大，但性能显著提高。我们在众所周知的 GLUE、SQuAD 和 RACE 基准上建立了新的SOAT结果。具体来说，我们将 RACE 精度提升至 89.4%，GLUE 基准提到 89.4，将 SQuAD 2.0 F1 分数提升到 92.2  。

# 3. 相关工作

## 3.1 扩大自然语言的表示学习

自然语言的学习表表示已被证明对广泛的 NLP 任务有用，并已被广泛采用。 (Mikolov et al., 2013; Le & Mikolov, 2014; Dai & Le, 2015; Peters et al., 2018; Devlin et al., 2019; Radford et al., 2018; 2019).这两年最显著的变化之一就是从预训练词嵌入，标准的（(Mikolov et al., 2013; Pennington et al., 2014）或者上下文的（McCann et al., 2017; Peters et al., 2018）到全网络预训练，然后根据具体任务进行微调（g (Dai & Le, 2015; Radford et al., 2018; Devlin et al., 2019). ）。这些工作表明更大的模型能获得更好的效果。

由于计算限制，特别是GPU/TPU的内存限制，大模型难以实验。目前的SOAT模型经常有几亿、几十亿参数，很容易抵达内存上限。为了解决这个问题，, Chen et al. (2016)提出一种梯度检查点(gradient checkpoint)方法来减少内存要求。.Gomez 等人 （2017） 提出了一种从下一层重建每个层的激活值的方法，以便它们不需要存储中间激活值。两种方法都以速度成本降低内存消耗。Raffel等人（2019年）建议使用模型并行化来训练一个巨大的模型。相比之下，我们的参数减少技术可减少内存消耗并提高训练速度。

## 3.2 跨层参数共享

transformer中有对跨层参数共享进行探索，但主要侧重于对标准encoder-decoder任务而不是预训练/微调任务。与我们的观察结果不同，,Dehghani et al. (2018) 表明跨层参数共享的网络(Universal Transformer,UT)在语言建模和subject-verb agreement上比在标准transformer上有着更好的性能。最近， Bai et al. (2019) 提出了transformer网络的深度均衡模型（Deep Equilibrium Model，DQE），表明DQE可以达到一个平衡点，而某一层的输入嵌入和输出嵌入保持不变。我们的观察表明，我们的嵌入是震荡的，而不是收敛的。Hao et al.(2019)将参数共享transformer与标准transformer相结合，进一步增加了标准transformer的参数量。

## 3.3 SENTENCE ORDERING OBJECTIVES

ALBERT 使用基于预测连续两段文本的顺序的损失函数来进行预训练。几位研究人员已经试验了与话语连贯性类似的预训练目标。话语中的连贯性和凝聚力得到了广泛的研究，并发现了许多连接相邻文本段的现象 (Hobbs, 1979; Halliday & Hasan, 1976; Grosz et al., 1995)。实践中发现那些有效的目标函数通常都是简单的。 Skipthought (Kiros et al., 2015) and FastSent (Hill et al., 2016)通过使用句子编码来预测相邻句子的单词来学习句子嵌入(sentence embedding)。.句子嵌入学习的其他目标包括预测更远的句子和预测明确的话语标记（Jernite等人，2017年;Nie等人，2019年），而不仅仅是相邻句子（Gan等人，2017年）。我们使用的损失函数与f Jernite et al. (2017)使用的句子顺序目标相似，在句子顺序目标中，通过确定连续两个句子的顺序来学习句子嵌入。然而，与上述大部分工作不同，我们的损失是用文本片段而不是句子来定义的。BERT的损失是预测句对中的第二个句子是否来自其它文档。我们在实验中比较了这种损失，发现句子排序是一项更具挑战性的预训任务，对某些下游任务更有用。同时，Wang等人（2019年）也试图预测连续两段文本的顺序，但它们在三向分类任务中将其与原始的下一句预测相结合，而不是通过经验比较两者。



# 4.ALBERT模型

## 4.1 模型架构选择

ALBERT模型的结构跟BERT相似，使用带GELU非线性的transformer encoder。词汇嵌入大小为$E$,encoder层数为$L$，hidden size为$H$. feed-forward/filter size为$4H$,attention head的数量为$H/64$。模型核心包括三点：

* Factorized embedding parameterization

  在BERT以及随后的预训练模型如XLNet和RoBERTa中，WordPiece embedding 的大小$E$与hidden size$是绑定的，即$E=H$。这中处理方法对于建模和实际情况来看并非最优的。原因如下：

  * 从建模的角度来看，WordPiece embedding旨在学习上下文独立表示，而隐藏层(hidden-layer)嵌入旨在学习上下文相关表示。正如上下文长度(context length)的实验表明的（Liu等人，2019年），像BERT这样的预训练模型所具有的强大表示能力来自于使用上下文为学习这种上下文相关表示提供信号。因此，从隐藏层大小 H 中取消 WordPiece 嵌入大小 E 的绑定使我们能够更高效地使用总模型参数，也就是可以让$H \gg E $。
  * 从实际角度看，自然语言处理中，词汇量$V$通常很大，如果$E \equiv H$,那么$H$的增大会导致词嵌入矩阵$V \times E$的增大，从而导致模型参数的剧增。因此，我们对$E$和$H$进行了解绑，对原来的词嵌入矩阵做了分解：从原来的$O(V \times H)$ 变为$O(V \times E + E \times H)$，如果$H \gg E$，参数量的减少是显著的。对于所有的WordPiece使用同样的embedding size $E$,因为相较于整词(whole-word)嵌入它们在文档中是更加均匀分布的。在有些论文中(Grave et al. (2017); Baevski & Auli (2018); Dai et al. (2019),它们也使用不同的embedding size。

* Cross-layer parameter sharing.

  在ALBERT中，我们提出了跨层参数共享来改善参数效率。参数共享的方式有多种，比如仅跨层共享前馈层网络(FFN)参数、仅共享注意力(attention)参数。ALBERT默认跨层共享所有参数。除非特指，所有的实验基于默认的参数共享方式。

  针对transformer网络，Dehghani et al. (2018) (Universal Transformer, UT) 和 Bai et al. (2019) (Deep Equilibrium Models, DQE) 使用了相同的策略进行探索。与我们的观察不同，Dehghani et al. (2018) 表明UT的性能优于vanilla Transformer。Bai et al.(2019) 表明，对于某层输入和输出嵌入保持不变的情况下，它们的DQE达到了平衡点。使用$L2$距离和余弦相似度衡量表明 我们的嵌入是振荡的而不是收敛的。

  ![image-20210109120605843](https://gitee.com/chengbo123/images/raw/master/image-20210109120605843.png)

  图1显示使用BERT-large和ALBERT-large,每层输入和输出嵌入的$L2$距离和余弦相似度。ALBERT层之间的转移比BERT更加平滑。这些结果表明权重共享对于稳定网络参数有效。与BERT相比，两者都有下降，但是即使在24层之后，他们也不会收敛到0。表明ALBERT参数的解空间与DQE的解空间大不相同。

* Inter-sentence coherence loss.

  除了MLM(masked language modeling)损失，BERT还使用了NSP(next-sentence prediction)损失。NSP损失是一个二分类损失，用于预测两个片段是否在原始文本中连续出现。正样本通过从训练语料中选择两个连续的片段，负样本来自不同的文档。NSP目标旨在改善下游任务性能，如自然语言推理任务。然而，后续的研究发现NSP的没有什么效果，因此去除了NSP。

  我们推测NSP无效的原因是与MLM相比没什么难度，如上所述，NSP将主题预测(topic prediction)和连贯性预测(coherence prediction)在单个任务中混为一谈。然而，主题预测相对于连贯性预测更容易，而且，与MLM目标所学到的东西有重叠。

  我们认为句子间建模(inter-sentence modeling)是语言理解的一个重要方面，但我们主要基于连贯性提出损失。ALBERT使用句子顺序预测(sentence order prediction,SOP)损失，避免了主题预测，侧重于句子间的建模连贯性。SOP损失中正样本的选择跟BERT一样来自于同一个文档的两个片段，而负样本直接把两个连续的片段顺序交换。这迫使模型能够进行更细粒度的学习。事实证明 NSP 根本无法解决 SOP 任务（即，它最终学习了更容易的主题预测信号，并在 SOP 任务上以随机基线水平执行），而 SOP 可以将 NSP 任务解决为合理的程度。ALBERT能够提高多句编码任务的下游任务性能。

  ## 4.2 模型设置

  ![image-20210109170727253](https://gitee.com/chengbo123/images/raw/master/image-20210109170727253.png)

同样的配置，ALBERT的参数量远小于BERT。

# 5. 实验结果

## 5.1 实验设置

* 语料：BOOkCORPUS和English Wikipedia，16GB。

* 格式："[CLS] $x_1$ [SEP] $x_2$ [SEP]",$x1,x_2$分别表示两个句子片段。

* 最大句长：512，随机生成10%的输入句子长度小于512.

* 词汇表大小：30000，使用SentencePience处理。

* 使用$n-gram$masking来生成MLM目标的masked输入，每个$n-gram$mask的长度随机选择。长度$n$的概率为：

  ![image-20210109171549099](https://gitee.com/chengbo123/images/raw/master/image-20210109171549099.png)

$n-gram$的最大长度为3（如MLM的目标由一个3-gram的完整词组成，例如"White House correspondents"）

* batch size=4096; optimizer为LAMB，学习率为0.00176.除非特指，所有的模型在TPU V3上训练125000steps.TPU的数量在64-512之间，根据模型大小而定。

## 5.2 评价基准

### 5.2.1  INTRINSIC EVALUATION

基于SQuAD和RACE数据集创建了一个新的数据集来评估MLM和句子分类任务的accuracy。近使用该数据集类检查模型如何收敛，不会影响到下游任务的评估。

### 5.2.2DOWNSTREAM EVALUATION

使用下列三个基准来评估模型：The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018),two versions of the Stanford Question Answering Dataset (SQuAD; Rajpurkar et al., 2016; 2018),and the ReAding Comprehension from Examinations (RACE) dataset (Lai et al., 2017).

### 5.3 OVERALL COMPARISON BETWEEN BERT AND ALBERT

![image-20210109173108050](https://gitee.com/chengbo123/images/raw/master/image-20210109173108050.png)

ALBERT-xxlarge的参数仅为BERT-large的70%,并且性能显著改进：

![image-20210109173249879](https://gitee.com/chengbo123/images/raw/master/image-20210109173249879.png)

ALBERT相对BERT具有更到的数据吞吐量。与BERT-large相比，ALBERT-large的速度是它的1.7倍，ALBERT-xxlarge则是它的0.3倍，因为模型更大。

## 5.4 FACTORIZED EMBEDDING PARAMETERIZATION

![image-20210109173653018](https://gitee.com/chengbo123/images/raw/master/image-20210109173653018.png)

* 不共享参数

  更大的embedding size具有更好的性能，但也不是越大越好。

* 共享参数

  embedding size=128效果最好。因此使用该参数做进一步的实验。

## 5.5 CROSS-LAYER PARAMETER SHARING

![image-20210109174206191](https://gitee.com/chengbo123/images/raw/master/image-20210109174206191.png)

## 5.6 SENTENCE ORDER PREDICTION (SOP)

![image-20210109174321612](https://gitee.com/chengbo123/images/raw/master/image-20210109174321612.png)

## 5.7 WHAT IF WE TRAIN FOR THE SAME AMOUNT OF TIME?

![image-20210109174419416](https://gitee.com/chengbo123/images/raw/master/image-20210109174419416.png)

## 5.8 ADDITIONAL TRAINING DATA AND DROPOUT EFFECTS

![image-20210109174506963](https://gitee.com/chengbo123/images/raw/master/image-20210109174506963.png)

![image-20210109174548157](https://gitee.com/chengbo123/images/raw/master/image-20210109174548157.png)

![image-20210109174609430](https://gitee.com/chengbo123/images/raw/master/image-20210109174609430.png)

## 5.9 CURRENT STATE-OF-THE-ART ON NLU TASKS

![image-20210109174654178](https://gitee.com/chengbo123/images/raw/master/image-20210109174654178.png)

![image-20210109174721235](https://gitee.com/chengbo123/images/raw/master/image-20210109174721235.png)