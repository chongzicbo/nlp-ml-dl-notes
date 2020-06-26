## ● Word2Vec中skip-gram是什么,Negative Sampling怎么做

### 参考回答：

Word2Vec通过学习文本然后用词向量的方式表征词的语义信息,然后使得语义相似的单词在嵌入式空间中的距离很近。而在Word2Vec模型中有Skip-Gram和CBOW两种模式,Skip-Gram是给定输入单词来预测上下文,而CBOW与之相反,是给定上下文来预测输入单词。Negative Sampling是对于给定的词,并生成其负采样词集合的一种策略,已知有一个词,这个词可以看做一个正例,而它的上下文词集可以看做是负例,但是负例的样本太多,而在语料库中,各个词出现的频率是不一样的,所以在采样时可以要求高频词选中的概率较大,低频词选中的概率较小,这样就转化为一个带权采样问题,大幅度提高了模型的性能。

## ● .FastText和Glovec原理

### 参考回答：

FastText是将句子中的每个词通过一个lookup层映射成词向量,对词向量叠加取平均作为句子的向量,然后直接用线性分类器进行分类,FastText中没有非线性的隐藏层,结构相对简单而且模型训练的更快。

Glovec融合了矩阵分解和全局统计信息的优势,统计语料库的词-词之间的共现矩阵,加快模型的训练速度而且又可以控制词的相对权重。

## ● word2vec实施过程

### 参考回答：

词向量其实是将词映射到一个语义空间，得到的向量。而word2vec是借用神经网络的方式实现的，考虑文本的上下文关系，有两种模型CBOW和Skip-gram，这两种模型在训练的过程中类似。Skip-gram模型是用一个词语作为输入，来预测它周围的上下文，CBOW模型是拿一个词语的上下文作为输入，来预测这个词语本身。

词向量训练的预处理步骤：

1.对输入的文本生成一个词汇表，每个词统计词频，按照词频从高到低排序，取最频繁的V个词，构成一个词汇表。每个词存在一个one-hot向量，向量的维度是V，如果该词在词汇表中出现过，则向量中词汇表中对应的位置为1，其他位置全为0。如果词汇表中不出现，则向量为全0

2.将输入文本的每个词都生成一个one-hot向量，此处注意保留每个词的原始位置，因为是上下文相关的

3.确定词向量的维数N

Skip-gram处理步骤：

1.确定窗口大小window，对每个词生成2*window个训练样本，(i, i-window)，(i, i-window+1)，...，(i, i+window-1)，(i, i+window)

2.确定batch_size，注意batch_size的大小必须是2*window的整数倍，这确保每个batch包含了一个词汇对应的所有样本

3.训练算法有两种：层次Softmax和Negative Sampling

4.神经网络迭代训练一定次数，得到输入层到隐藏层的参数矩阵，矩阵中每一行的转置即是对应词的词向量

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552911465006_431DE2BC26D651525012526447CDCBE5)

CBOW的处理步骤：

1.确定窗口大小window，对每个词生成2*window个训练样本，(i-window, i)，(i-window+1, i)，...，(i+window-1, i)，(i+window, i)

2.确定batch_size，注意batch_size的大小必须是2*window的整数倍，这确保每个batch包含了一个词汇对应的所有样本

3.训练算法有两种：层次Softmax和Negative Sampling

4.神经网络迭代训练一定次数，得到输入层到隐藏层的参数矩阵，矩阵中每一行的转置即是对应词的词向量

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552911440225_8CC56AE3586EFD495C70745D5E7A5B0A)

参数矩阵解释：

对输入层到隐藏层的参数包含W和b，我们需要的是W，这里的W是一个矩阵，shape=(N,V)。其中V是上文所述的词表的大小，N是需要生成的词向量的维数。N同样也是隐藏层（第一层）中的隐藏节点个数。

每次一个batch_size输入其实一个矩阵(batch_size, V)，记为X，隐藏层输出为Y，公式为![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552911565798_04E861EBB1A1C5A8143A01A9E72E9AAE)。所有的输入共享一个W，每次迭代的时候都在修改W，由于one-hot的性质，每次修改W只修改1对应的那一行。而这一行也就是词向量（转置后）

神经网络像是一个黑盒子，这其中的概念很难理解，这里给出我对词向量训练的个人理解：对于每个词s，训练数据对应的标记是另一个词t，训练其实是想找到一种映射关系，让s映射到t。但很显然我们不是希望找到一个线性函数，使得给定s一定能得到t，我们希望的是能够通过s得到一类词T，包含t。对于T中的每个t，由于在s上下文中出现的频次不同，自然能得到一个概率，频次越高说明s与t相关性越高。

对于词向量，或者说参数矩阵W，可以认为是一个将词映射到语义空间的桥梁，s与t相关性越高，则认为其在语义空间中越近，那么对应的桥梁也越靠近。如果用向量来理解的话就是向量之前的夹角越小，我们使用向量来表示这个词的信息，重要的是得到了语义信息。在实际应用中，生成一段文本，我们可以判断词与词的向量之间相似度，如果过低则就需要怀疑是否正确了。

## ● softmax的原理了解

### 参考回答：

考虑一个多分类问题，即预测变量y可以取k个离散值中的任何一个.比如一个邮件分类系统将邮件分为私人邮件，工作邮件和垃圾邮件。由于y仍然是一个离散值，只是相对于二分类的逻辑回归多了一些类别。下面将根据多项式分布建模。

考虑将样本共有k类，每一类的概率分别为![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912332812_49DEA3A986F227D201CCC75A9F898196),由于![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912316380_DBF51BEB257F892842BFE9D8EA63B67A)，所以通常我们只需要k-1个参数![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912296547_51EBF30C85A21C7DEE5E7210782179B9)即可

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912277088_72EF8A8B451D26666AF2B0948D72B8B7)，![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912263846_04DBFCB5E6C1EFE47914374A2FD6E31B)

为了推导，引入表达式：

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912251158_C962D59D15587B82F9369EE90B9F5F0E)

上面T(y)是k-1维列向量,其中y = 1, 2, ...k.

T(y)i 表示向量T(y)的第i个元素。

还要引入表达式![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912237198_90DF233EC01F187D2C39849CF206BE7D),如果大括号里面为真，则真个表达式就为1，否则为0.例如：1{2=3} = 0和1{3=3} = 1.

则上面的k个向量就可以表示为![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912218885_2FB3BE5FD28C64BE8F3722D9ADA39CD5)

以为y只能属于某一个类别，于是T(y)中只能有一个元素为1其他元素都为0，可以求出k-1个元素的期望：![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912204968_040957ECCA47B5DBA52C4AD11631D00C)

定义：![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912191552_EDDC1007CDBC89BF6FC0FFF61FBC5173)

其中i = 1,2,...k.则有：

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912178118_8A678117375294A622DF9EF02883AEC6)

也就容易得出：![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912162976_0392BA677073060A89D408E926ABAED9)，由该式和上面使得等式：![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912142676_7D5FE97B6B95A23709D7BA87566D8173)一起可以得到：![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912020684_7230EFBD6A12F70A5C4AE71A61421A15)这个函数就是softmax函数。

然后假设![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912001310_B469096FC45F5EE52E99FD0AE9BF49F8)和![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552911987122_5324B720996E57EEC63F94FDF3155C0B)具有线性关系，即![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552911965946_60229DCD5261960716C722C07147A653)

于是从概率的角度出发：

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552911950771_AE7E660593807F2457EC2DEE2DE93E13)

其中![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552911933261_2859076CD1212DAD72F6C04A67F723D8)这个模型就是softmax回归（softmax regression）,它是逻辑回归的泛化。

这样我们的输出：

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552911911951_FCE07DCC5BB5009B1504A8568F025CCA)

就是输出了x属于（1,2,...k-1）中每一类的概率，当然属于第k类的概率就是：![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552911897289_9B9FCDA45372DC22697AC68B771918EF)

下面开始拟合参数

同样使用最大化参数θ的对数似然函数：

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552911884751_B46F06E75A30F53AF5C22717DF0D38C2)

这里使用梯度下降和牛顿法均可。

## ● Wod2vec公式

### 参考回答：

Hierarchical Softmax

CBOW

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912495064_37DC3B534BD797526EF8D0B184FF403A)

Skip-gram

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912476101_B9D13BDA2F32C3ACFDF7E502849D91D4)

Negative Sampling

CBOW

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912462142_E47B0BF57ACE7536E37AAFDB8B472411)

Skip-gram

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912436853_50F07E2C5E8D1143BC6E34CC2F5B1A19)![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912417503_A31401F22D2E19BEC8B9FF5C4656C808)

 

## ● Wod2vec公式

### 参考回答：

Hierarchical Softmax

CBOW

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912635249_17EC4D1F6504574470EC6D2D263BEAF1)

Skip-gram

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912618843_94673AC62EF75721AA3CF709CAD48215)

Negative Sampling

CBOW

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912602945_658F78C975565AC645AE32499F1B7065)

Skip-gram

![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912587275_570B7B7FA973AF3A033FEC4CAAC88D1A)![img](https://uploadfiles.nowcoder.com/images/20190318/311436_1552912569796_4CC4109FCFDDAE370DD4C20BE04884F1)

 

## ● 使用gensim的word similar方法预测句子

### 参考回答：

利用gensim训练Word2vec向量，得到词向量空间，通过词向量空间预测词之间的相似度，从而去预测由词组成的句子之间的相似度。