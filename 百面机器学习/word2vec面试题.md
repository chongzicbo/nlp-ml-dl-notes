## 算法十问

1. 在小数据集中，Skip-gram和CBOW哪种表现更好？

   > Skip-gram是用一个center word预测其context里的word；而CBOW是用context里的所有word去预测一个center word。显然，前者对训练数据的利用更高效（构造的数据集多），因此，对于较小的语料库，Skip-gram是更好的选择。

2. 为什么要使用HS（Hierarchical Softmax ）和负采样（Negative Sampling）？

   > 两个模型的原始做法都是做内积，经过 Softmax 后得到概率，因此复杂度很高。假设我们拥有一个百万量级的词典，每一步训练都需要计算上百万次词向量的内积，显然这是无法容忍的。因此人们提出了两种较为实用的训练技巧，即HS和Negative Sampling。

3. 介绍一下HS（Hierarchical Softmax ）

   > HS 是试图用词频建立一棵哈夫曼树，那么经常出现的词路径会比较短。树的叶子节点表示词，共词典大小多个，而非叶子结点是模型的参数，比词典个数少一个。**要预测的词，转化成预测从根节点到该词所在叶子节点的路径，是多个二分类问题。本质是把 N 分类问题变成 log(N)次二分类**

4. 介绍一下负采样（Negative Sampling）

   > 把原来的 Softmax 多分类问题，直接转化成一个正例和多个负例的二分类问题。让正例预测 1，负例预测 0，这样子更新局部的参数。

5. 负采样为什么要用词频来做采样概率？

> 可以让频率高的词先学习，然后带动其他词的学习。

1. 对比 Skip-gram 和 CBOW

   > **CBOW 会比Skip-gram训练速度更快**，因为前者每次会更新 context(w) 的词向量，而 Skip-gram 只更新核心词的词向量。 **Skip-gram 对低频词效果比 CBOW好**，因为Skip-gram 是尝试用当前词去预测上下文，当前词是低频词还是高频词没有区别。但是 CBOW 相当于是**完形填空**，会选择最常见或者说概率最大的词来补全，因此不太会选择低频词。

2. 对比字向量和词向量

   > **字向量可以解决未登录词的问题**，以及可以**避免分词**；词向量包含的**语义空间更大，更加丰富**，如果语料足够的情况下，词向量是能够学到更多的语义信息。

3. 如何衡量word2vec得出的词/字向量的质量？

   > 在实际工程中一般以**word embedding对于实际任务的收益为评价标准**，包括**词汇类比任务**（如king – queen = man - woman）以及NLP中常见的应用任务，比如**命名实体识别（NER），关系抽取（RE）**等。

4. 神经网络框架里的Embedding层和word-embedding有什么关系？

   > Embedding层就是以one hot为输入（实际一般输入字或词的id）、中间层节点为字向量维数的全连接层。而这个全连接层的参数，就是一个“字向量表”，即word-embedding。

5. word2vec的缺点？

   > **没有考虑词序**，因为它**假设了词的上下文无关(把概率变为连乘)**；**没有考虑全局的统计信息**。

## 面试真题

1. 为什么训练得到的字词向量会有如下一些性质，比如向量的夹角余弦、向量的欧氏距离都能在一定程度上反应字词之间的相似性？

   > 我们在用语言模型无监督训练时，是开了窗口的，通过前n个字预测下一个字的概率，这个n就是窗口的大小，同一个窗口内的词语，会有相似的更新，这些更新会累积，而具有相似模式的词语就会把这些相似更新累积到可观的程度。

2. word2vec跟Glove的异同？

   GloVe与word2vec，两个模型都可以根据词汇的“共现co-occurrence”信息，将词汇编码成一个向量（所谓共现，即语料中词汇一块出现的频率）。**两者最直观的区别在于，word2vec是“predictive”的模型，而GloVe是“count-based”的模型**。具体是什么意思呢？

   

   Predictive的模型，如Word2vec，根据context预测中间的词汇，要么根据中间的词汇预测context，分别对应了word2vec的两种训练方式cbow和skip-gram。对于word2vec，采用三层神经网络就能训练，最后一层的输出要用一个Huffuman树进行词的预测（这一块有些大公司面试会问到，为什么用Huffuman树，大家可以思考一下）。

   ![img](https://pic2.zhimg.com/v2-2643257651782dfb401527fa016d06e1_r.jpg)

   Count-based模型，如GloVe，本质上是对共现矩阵进行降维。首先，构建一个词汇的共现矩阵，每一行是一个word，每一列是context。共现矩阵就是计算每个word在每个context出现的频率。由于context是多种词汇的组合，其维度非常大，我们希望像network embedding一样，在context的维度上降维，学习word的低维表示。这一过程可以视为共现矩阵的重构问题，即reconstruction loss。(这里再插一句，降维或者重构的本质是什么？我们选择留下某个维度和丢掉某个维度的标准是什么？Find the lower-dimensional representations which can explain most of the variance in the high-dimensional data，这其实也是PCA的原理)。

   

   两种方法都能学习词的向量表示，在real world application中，他们效果上有啥差别呢？

   答案是performance上差别不大。

   

   两个模型在并行化上有一些不同，即GloVe更容易并行化，所以对于较大的训练数据，GloVe更快。

   

   在英文上，[glove](https://radimrehurek.com/gensim/models/word2vec.html) for GloVe 和 [gensim](https://github.com/maciejkula/glove-python) for word2vec是常用的训练词向量的python package，完全可以使用自己的训练语料训练词向量。当然，他们都提供了google news（英文）上训练好的词向量，大家完全可以下载下来，直接使用。对于中文的训练语料，可以使用[sogou中文新闻语料](http://www.sogou.com/labs/resource/list_news.php)。

   

   更多关于GloVe和word2vec的区别详见论文：

   [Don’t count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors](http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf)

   

   本文翻译自[Stephan Gouws](https://www.quora.com/profile/Stephan-Gouws) from Quora，加入了一点点自己的思考。

   当然还是希望大家读一读原答案：https://www.quora.com/How-is-GloVe-different-from-word2vec

3. word2vec 相比之前的 Word Embedding 方法好在什么地方

4. #### 为什么基于skip-gram的word2vec在低频词汇相比cbow更有效？

CBOW是根据上下文预测当中的一个词，也就是用多个词预测一个词

比如这样一个句子yesterday was really a [...] day，中间可能是good也可能是nice，比较生僻的词是delightful。当CBOW去预测中间的词的时候，它只会考虑模型最有可能出现的结果，比如good和nice，生僻词delightful就被忽略了。

而对于[...] was really a delightful day这样的句子，每个词在进入模型后，都相当于进行了均值处理（权值乘以节点），delightful本身因为生僻，出现得少，所以在进行加权平均后，也容易被忽视。

Skip-Gram是根据一个词预测它的上下文，也就是用一个词预测多个词，每个词都会被单独得训练，较少受其他高频的干扰。所以对于生僻词Skip-Gram的word2vec更占优。

由于 CBOW 需要更多的数据，所以它对高频词汇更敏感，从而在低频词汇上表现没有 skip-gram 好。

## 参考资料

1. https://blog.csdn.net/zhangxb35/article/details/74716245
2. https://spaces.ac.cn/archives/4122