# 1.RNN为什么具有记忆功能

RNN具有递归效应，上一时刻的隐层状态参与到这个时刻的计算过程中。

# 2.RNN原理

## 2.1 不含隐藏状态的神经网络

考虑一个含单隐藏层的多层感知机。给定样本数为$n$、输入个数(特征数或者特征向量维度)为$d$的小批量数据样本$X \in R ^{n \times d}$。设隐藏层的激活函数为$\phi$，那么隐藏层的输出$H\in R^{n \times h}$计算为：
$$
H=\phi(XW_{xh}+b_h)
$$
其中隐藏层权重参数$W_{xh}\in R^{d \times h}$，隐藏层偏差参数$b_h \in R^{1 \times h}$，$h$为隐藏单元个数。把隐藏变量$H$作为输出层的输入,且设输出个数为$q$,则输出层的输出为：
$$
O=HW_{hq}+b_q
$$
其中输出变量$O\in R^{n \times q}$,输出层权重参数$W_{hq}\in R^{h \times q}$,输出层偏差参数$b_q \in R^{1 \times q}$。如果是分类问题，可以使用softmax（O）来计算输出类别的概率。

## 2.2 含隐藏状态的循环神经网络

现在我们考虑输入数据存在时间相关性的情况。假设$\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$是序列中时间步tt的小批量输入，$\boldsymbol{H}_t \in \mathbb{R}^{n \times h}$是该时间步的隐藏变量。与多层感知机不同的是，这里我们保存上一时间步的隐藏变量$H_{t−1}$，并引入一个新的权重参数$\boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$，该参数用来描述在当前时间步如何使用上一时间步的隐藏变量。具体来说，时间步$t$的隐藏变量的计算由当前时间步的输入和上一时间步的隐藏变量共同决定：
$$
\boldsymbol{H}_t = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}  + \boldsymbol{b}_h).
$$
与多层感知机相比，我们在这里添加了$\boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}$一项。由上式中相邻时间步的隐藏变量$H_t$和$H_{t−1}$之间的关系可知，这里的隐藏变量能够捕捉截至当前时间步的序列的历史信息，就像是神经网络当前时间步的状态或记忆一样。因此，该隐藏变量也称为隐藏状态。由于隐藏状态在当前时间步的定义使用了上一时间步的隐藏状态，上式的计算是循环的。使用循环计算的网络即循环神经网络（recurrent neural network）。

循环神经网络有很多种不同的构造方法。含上式所定义的隐藏状态的循环神经网络是极为常见的一种。若无特别说明，本章中的循环神经网络均基于上式中隐藏状态的循环计算。在时间步$t$，输出层的输出和多层感知机中的计算类似：
$$
\boldsymbol{O}_t = \boldsymbol{H}_t \boldsymbol{W}_{hq} + \boldsymbol{b}_q.
$$
循环神经网络的参数包括隐藏层的权重$\boldsymbol{W}_{xh} \in \mathbb{R}^{d \times h}$、$\boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$和偏差 $\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$，以及输出层的权重$\boldsymbol{W}_{hq} \in \mathbb{R}^{h \times q}$和偏差$\boldsymbol{b}_q \in \mathbb{R}^{1 \times q}$。值得一提的是，即便在不同时间步，循环神经网络也始终使用这些模型参数。因此，循环神经网络模型参数的数量不随时间步的增加而增长。

图6.1展示了循环神经网络在3个相邻时间步的计算逻辑。在时间步$t$，隐藏状态的计算可以看成是将输入$X_t$和前一时间步隐藏状态$H_{t−1}$连结后输入一个激活函数为$\phi$的全连接层。该全连接层的输出就是当前时间步的隐藏状态$H_t$，且模型参数为$W_{xh}$与$W_{hh}$的连结，偏差为$b_h$。当前时间步$t$的隐藏状态$H_t$将参与下一个时间步$t+1$的隐藏状态$H_{t+1}$的计算，并输入到当前时间步的全连接输出层。

![](https://zh.gluon.ai/_images/rnn.svg)

# 3. LSTM原理

## 3.1 长短期记忆

LSTM 中引入了3个门，即输入门（input gate）、遗忘门（forget gate）和输出门（output gate），以及与隐藏状态形状相同的记忆细胞（某些文献把记忆细胞当成一种特殊的隐藏状态），从而记录额外的信息。

## 3.2 输入门、遗忘门和输出门

与门控循环单元中的重置门和更新门一样，如图6.7所示，长短期记忆的门的输入均为当前时间步输入$X_t$与上一时间步隐藏状态$H_{t−1}$，输出由激活函数为sigmoid函数的全连接层计算得到。如此一来，这3个门元素的值域均为[0,1]。

![长短期记忆中输入门、遗忘门和输出门的计算](https://zh.gluon.ai/_images/lstm_0.svg)

<center>长短期记忆中输入门、遗忘门和输出门的计算<center>

具体来说，假设隐藏单元个数为$h$，给定时间步$t$的小批量输入$X_t \in R^{n×d}$（样本数为$n$，输入个数为$d$）和上一时间步隐藏状态$H_{t−1}\in R^{n×h}$。 时间步$t$的输入门$I_t\in R^{n×h}$、遗忘门$F_t\in R^{n×h}$和输出门$O_t\in R^{n×h}$分别计算如下：
$$
\begin{split}\begin{aligned}
\boldsymbol{I}_t &= \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xi} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hi} + \boldsymbol{b}_i),\\
\boldsymbol{F}_t &= \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xf} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hf} + \boldsymbol{b}_f),\\
\boldsymbol{O}_t &= \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xo} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{ho} + \boldsymbol{b}_o),
\end{aligned}\end{split}
$$
其中的$\boldsymbol{W}_{xi}, \boldsymbol{W}_{xf}, \boldsymbol{W}_{xo} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hi}, \boldsymbol{W}_{hf}, \boldsymbol{W}_{ho} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_i, \boldsymbol{b}_f, \boldsymbol{b}_o \in \mathbb{R}^{1 \times h}$是偏差参数。

## 3.3 候选记忆细胞

接下来，长短期记忆需要计算候选记忆细胞$\tilde{\boldsymbol{C}}_t$。它的计算与上面介绍的3个门类似，但使用了值域在$[−1,1]$的tanh函数作为激活函数，如图6.8所示。

![](https://zh.gluon.ai/_images/lstm_1.svg)

<center>长短期记忆中候选记忆细胞的计算<center>

具体来说，时间步$t$的候选记忆细胞$\tilde{\boldsymbol{C}}_t \in \mathbb{R}^{n \times h}$的计算为
$$
\tilde{\boldsymbol{C}}_t = \text{tanh}(\boldsymbol{X}_t \boldsymbol{W}_{xc} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hc} + \boldsymbol{b}_c),
$$
其中$\boldsymbol{W}_{xc} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hc} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_c \in \mathbb{R}^{1 \times h}$是偏差参数。



## 3.4 记忆细胞

我们可以通过元素值域在$[0,1]$的输入门、遗忘门和输出门来控制隐藏状态中信息的流动，这一般也是通过使用按元素乘法（符号为⊙）来实现的。当前时间步记忆细胞$\boldsymbol{C}_t \in \mathbb{R}^{n \times h}$的计算组合了上一时间步记忆细胞和当前时间步候选记忆细胞的信息，并通过遗忘门和输入门来控制信息的流动：
$$
\boldsymbol{C}_t = \boldsymbol{F}_t \odot \boldsymbol{C}_{t-1} + \boldsymbol{I}_t \odot \tilde{\boldsymbol{C}}_t.
$$
如图6.9所示，遗忘门控制上一时间步的记忆细胞$\boldsymbol{C}_{t-1}$中的信息是否传递到当前时间步，而输入门则控制当前时间步的输入$X_t$通过候选记忆细胞$\tilde{\boldsymbol{C}}_t$如何流入当前时间步的记忆细胞。如果遗忘门一直近似1且输入门一直近似0，过去的记忆细胞将一直通过时间保存并传递至当前时间步。这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。

![](https://zh.gluon.ai/_images/lstm_2.svg)

<center>长短期记忆中记忆细胞的计算。这里的⊙⊙是按元素乘法</center>

## 3.5 隐藏状态

有了记忆细胞以后，接下来我们还可以通过输出门来控制从记忆细胞到隐藏状态$\boldsymbol{H}_t \in \mathbb{R}^{n \times h}$的信息的流动：
$$
\boldsymbol{H}_t = \boldsymbol{O}_t \odot \text{tanh}(\boldsymbol{C}_t).
$$
这里的tanh函数确保隐藏状态元素值在-1到1之间。需要注意的是，当输出门近似1时，记忆细胞信息将传递到隐藏状态供输出层使用；当输出门近似0时，记忆细胞信息只自己保留。图6.10展示了长短期记忆中隐藏状态的计算。

![](https://zh.gluon.ai/_images/lstm_3.svg)

<center>长短期记忆中隐藏状态的计算。这里的 ⊙ 是按元素乘法</center>

# 4.门控循环神经网络(GRU）

上一节介绍了循环神经网络中的梯度计算方法。我们发现，当时间步数较大或者时间步较小时，循环神经网络的梯度较容易出现衰减或爆炸。虽然裁剪梯度可以应对梯度爆炸，但无法解决梯度衰减的问题。通常由于这个原因，循环神经网络在实际中较难捕捉时间序列中时间步距离较大的依赖关系。

门控循环神经网络（gated recurrent neural network）的提出，正是为了更好地捕捉时间序列中时间步距离较大的依赖关系。它通过可以学习的门来控制信息的流动。其中，门控循环单元（gated recurrent unit，GRU）是一种常用的门控循环神经网络 [1, 2]。

## 4.1 门控循环单元

下面将介绍门控循环单元的设计。它引入了重置门（reset gate）和更新门（update gate）的概念，从而修改了循环神经网络中隐藏状态的计算方式。

### 4.1.1 重置门和更新门

如图6.4所示，门控循环单元中的重置门和更新门的输入均为当前时间步输入$\boldsymbol{X}_t$与上一时间步隐藏状态$\boldsymbol{H}_{t-1}$，输出由激活函数为sigmoid函数的全连接层计算得到。

![](https://zh.gluon.ai/_images/gru_1.svg)

<center>
    门控循环单元中重置门和更新门的计算
</center>

具体来说，假设隐藏单元个数为$h$，给定时间步$t$的小批量输入$X_t \in R^{n×d}$（样本数为$n$，输入个数为$d$）和上一时间步隐藏状态$H_{t−1} \in R^{n×h}$。重置门$R_t\in R^{n×h}$和更新门$Z_t \in R^{n×h}$的计算如下：
$$
\begin{split}\begin{aligned}
\boldsymbol{R}_t = \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xr} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hr} + \boldsymbol{b}_r),\\
\boldsymbol{Z}_t = \sigma(\boldsymbol{X}_t \boldsymbol{W}_{xz} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hz} + \boldsymbol{b}_z),
\end{aligned}\end{split}
$$
其中$\boldsymbol{W}_{xr}, \boldsymbol{W}_{xz} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hr}, \boldsymbol{W}_{hz} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_r, \boldsymbol{b}_z \in \mathbb{R}^{1 \times h}$是偏差参数。[“多层感知机”](https://zh.gluon.ai/chapter_deep-learning-basics/mlp.html)一节中介绍过，sigmoid函数可以将元素的值变换到0和1之间。因此，重置门$R_t$和更新门$Z_t$中每个元素的值域都是[0,1]。

### 4.1.2 候选隐藏状态

接下来，门控循环单元将计算候选隐藏状态来辅助稍后的隐藏状态计算。如图6.5所示，我们将当前时间步重置门的输出与上一时间步隐藏状态做按元素乘法（符号为⊙⊙）。如果重置门中元素值接近0，那么意味着重置对应隐藏状态元素为0，即丢弃上一时间步的隐藏状态。如果元素值接近1，那么表示保留上一时间步的隐藏状态。然后，将按元素乘法的结果与当前时间步的输入连结，再通过含激活函数tanh的全连接层计算出候选隐藏状态，其所有元素的值域为[−1,1]。

![](https://zh.gluon.ai/_images/gru_2.svg)

<center>
    门控循环单元中候选隐藏状态的计算。这里的 ⊙ 是按元素乘法
</center>

具体来说，时间步tt的候选隐藏状态$\tilde{\boldsymbol{H}}_t \in \mathbb{R}^{n \times h}$的计算为
$$
\tilde{\boldsymbol{H}}_t = \text{tanh}(\boldsymbol{X}_t \boldsymbol{W}_{xh} + \left(\boldsymbol{R}_t \odot \boldsymbol{H}_{t-1}\right) \boldsymbol{W}_{hh} + \boldsymbol{b}_h),
$$
其中$\boldsymbol{W}_{xh} \in \mathbb{R}^{d \times h}$和$\boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$是权重参数，$\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$是偏差参数。从上面这个公式可以看出，重置门控制了上一时间步的隐藏状态如何流入当前时间步的候选隐藏状态。而上一时间步的隐藏状态可能包含了时间序列截至上一时间步的全部历史信息。因此，重置门可以用来丢弃与预测无关的历史信息。

### 4.1.3 隐藏状态

最后，时间步$t$的隐藏状态$\boldsymbol{H}_t \in \mathbb{R}^{n \times h}$的计算使用当前时间步的更新门$Z_t$来对上一时间步的隐藏状态$H_{t−1}$和当前时间步的候选隐藏状态$\tilde{\boldsymbol{H}}_t$做组合：
$$
\boldsymbol{H}_t = \boldsymbol{Z}_t \odot \boldsymbol{H}_{t-1}  + (1 - \boldsymbol{Z}_t) \odot \tilde{\boldsymbol{H}}_t.
$$
![](https://zh.gluon.ai/_images/gru_3.svg)

<center>
    门控循环单元中隐藏状态的计算。这里的 ⊙ 是按元素乘法
</center>

值得注意的是，更新门可以控制隐藏状态应该如何被包含当前时间步信息的候选隐藏状态所更新，如图6.6所示。假设更新门在时间步$t′$到$t（t′<t）$之间一直近似1。那么，在时间步$t′$到$t$之间的输入信息几乎没有流入时间步$t$的隐藏状态$H_t$。实际上，这可以看作是较早时刻的隐藏状态$H_{t′−1}$一直通过时间保存并传递至当前时间步$t$。这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时间序列中时间步距离较大的依赖关系。

我们对门控循环单元的设计稍作总结：

- 重置门有助于捕捉时间序列里短期的依赖关系；
- 更新门有助于捕捉时间序列里长期的依赖关系。



# 5. RNN梯度消失的原因

经典的RNN结构如下图所示：

![](https://pic2.zhimg.com/v2-37ba7e208c51e0a5bfd37b60da074b79_r.jpg)

假设我们的时间序列只有三段， ![[公式]](https://www.zhihu.com/equation?tex=S_%7B0%7D) 为给定值，神经元没有激活函数，则RNN最简单的前向传播过程如下：

![[公式]](https://www.zhihu.com/equation?tex=S_%7B1%7D%3DW_%7Bx%7DX_%7B1%7D%2BW_%7Bs%7DS_%7B0%7D%2Bb_%7B1%7D)![[公式]](https://www.zhihu.com/equation?tex=O_%7B1%7D%3DW_%7Bo%7DS_%7B1%7D%2Bb_%7B2%7D)

![[公式]](https://www.zhihu.com/equation?tex=S_%7B2%7D%3DW_%7Bx%7DX_%7B2%7D%2BW_%7Bs%7DS_%7B1%7D%2Bb_%7B1%7D)![[公式]](https://www.zhihu.com/equation?tex=O_%7B2%7D%3DW_%7Bo%7DS_%7B2%7D%2Bb_%7B2%7D)

![[公式]](https://www.zhihu.com/equation?tex=S_%7B3%7D%3DW_%7Bx%7DX_%7B3%7D%2BW_%7Bs%7DS_%7B2%7D%2Bb_%7B1%7D)![[公式]](https://www.zhihu.com/equation?tex=O_%7B3%7D%3DW_%7Bo%7DS_%7B3%7D%2Bb_%7B2%7D)

假设在t=3时刻，损失函数为 ![[公式]](https://www.zhihu.com/equation?tex=L_%7B3%7D%3D%5Cfrac%7B1%7D%7B2%7D%28Y_%7B3%7D-O_%7B3%7D%29%5E%7B2%7D) 。

则对于一次训练任务的损失函数为 ![[公式]](https://www.zhihu.com/equation?tex=L%3D%5Csum_%7Bt%3D0%7D%5E%7BT%7D%7BL_%7Bt%7D%7D) ，即每一时刻损失值的累加。

使用随机梯度下降法训练RNN其实就是对 ![[公式]](https://www.zhihu.com/equation?tex=W_%7Bx%7D+) 、 ![[公式]](https://www.zhihu.com/equation?tex=W_%7Bs%7D) 、 ![[公式]](https://www.zhihu.com/equation?tex=W_%7Bo%7D) 以及 ![[公式]](https://www.zhihu.com/equation?tex=b_%7B1%7D)![[公式]](https://www.zhihu.com/equation?tex=b_%7B2%7D) 求偏导，并不断调整它们以使L尽可能达到最小的过程。

现在假设我们我们的时间序列只有三段，t1，t2，t3。

我们只对t3时刻的 ![[公式]](https://www.zhihu.com/equation?tex=W_%7Bx%7D%E3%80%81W_%7Bs%7D%E3%80%81W_%7B0%7D) 求偏导（其他时刻类似）：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%7BL_%7B3%7D%7D%7D%7B%5Cpartial%7BW_%7B0%7D%7D%7D%3D%5Cfrac%7B%5Cpartial%7BL_%7B3%7D%7D%7D%7B%5Cpartial%7BO_%7B3%7D%7D%7D%5Cfrac%7B%5Cpartial%7BO_%7B3%7D%7D%7D%7B%5Cpartial%7BW_%7Bo%7D%7D%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%7BL_%7B3%7D%7D%7D%7B%5Cpartial%7BW_%7Bx%7D%7D%7D%3D%5Cfrac%7B%5Cpartial%7BL_%7B3%7D%7D%7D%7B%5Cpartial%7BO_%7B3%7D%7D%7D%5Cfrac%7B%5Cpartial%7BO_%7B3%7D%7D%7D%7B%5Cpartial%7BS_%7B3%7D%7D%7D%5Cfrac%7B%5Cpartial%7BS_%7B3%7D%7D%7D%7B%5Cpartial%7BW_%7Bx%7D%7D%7D%2B%5Cfrac%7B%5Cpartial%7BL_%7B3%7D%7D%7D%7B%5Cpartial%7BO_%7B3%7D%7D%7D%5Cfrac%7B%5Cpartial%7BO_%7B3%7D%7D%7D%7B%5Cpartial%7BS_%7B3%7D%7D%7D%5Cfrac%7B%5Cpartial%7BS_%7B3%7D%7D%7D%7B%5Cpartial%7BS_%7B2%7D%7D%7D%5Cfrac%7B%5Cpartial%7BS_%7B2%7D%7D%7D%7B%5Cpartial%7BW_%7Bx%7D%7D%7D%2B%5Cfrac%7B%5Cpartial%7BL_%7B3%7D%7D%7D%7B%5Cpartial%7BO_%7B3%7D%7D%7D%5Cfrac%7B%5Cpartial%7BO_%7B3%7D%7D%7D%7B%5Cpartial%7BS_%7B3%7D%7D%7D%5Cfrac%7B%5Cpartial%7BS_%7B3%7D%7D%7D%7B%5Cpartial%7BS_%7B2%7D%7D%7D%5Cfrac%7B%5Cpartial%7BS_%7B2%7D%7D%7D%7B%5Cpartial%7BS_%7B1%7D%7D%7D%5Cfrac%7B%5Cpartial%7BS_%7B1%7D%7D%7D%7B%5Cpartial%7BW_%7Bx%7D%7D%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%7BL_%7B3%7D%7D%7D%7B%5Cpartial%7BW_%7Bs%7D%7D%7D%3D%5Cfrac%7B%5Cpartial%7BL_%7B3%7D%7D%7D%7B%5Cpartial%7BO_%7B3%7D%7D%7D%5Cfrac%7B%5Cpartial%7BO_%7B3%7D%7D%7D%7B%5Cpartial%7BS_%7B3%7D%7D%7D%5Cfrac%7B%5Cpartial%7BS_%7B3%7D%7D%7D%7B%5Cpartial%7BW_%7Bs%7D%7D%7D%2B%5Cfrac%7B%5Cpartial%7BL_%7B3%7D%7D%7D%7B%5Cpartial%7BO_%7B3%7D%7D%7D%5Cfrac%7B%5Cpartial%7BO_%7B3%7D%7D%7D%7B%5Cpartial%7BS_%7B3%7D%7D%7D%5Cfrac%7B%5Cpartial%7BS_%7B3%7D%7D%7D%7B%5Cpartial%7BS_%7B2%7D%7D%7D%5Cfrac%7B%5Cpartial%7BS_%7B2%7D%7D%7D%7B%5Cpartial%7BW_%7Bs%7D%7D%7D%2B%5Cfrac%7B%5Cpartial%7BL_%7B3%7D%7D%7D%7B%5Cpartial%7BO_%7B3%7D%7D%7D%5Cfrac%7B%5Cpartial%7BO_%7B3%7D%7D%7D%7B%5Cpartial%7BS_%7B3%7D%7D%7D%5Cfrac%7B%5Cpartial%7BS_%7B3%7D%7D%7D%7B%5Cpartial%7BS_%7B2%7D%7D%7D%5Cfrac%7B%5Cpartial%7BS_%7B2%7D%7D%7D%7B%5Cpartial%7BS_%7B1%7D%7D%7D%5Cfrac%7B%5Cpartial%7BS_%7B1%7D%7D%7D%7B%5Cpartial%7BW_%7Bs%7D%7D%7D)

可以看出对于 ![[公式]](https://www.zhihu.com/equation?tex=W_%7B0%7D) 求偏导并没有长期依赖，但是对于 ![[公式]](https://www.zhihu.com/equation?tex=W_%7Bx%7D%E3%80%81W_%7Bs%7D) 求偏导，会随着时间序列产生长期依赖。因为 ![[公式]](https://www.zhihu.com/equation?tex=S_%7Bt%7D) 随着时间序列向前传播，而 ![[公式]](https://www.zhihu.com/equation?tex=S_%7Bt%7D) 又是 ![[公式]](https://www.zhihu.com/equation?tex=W_%7Bx%7D%E3%80%81W_%7Bs%7D)的函数。

根据上述求偏导的过程，我们可以得出任意时刻对 ![[公式]](https://www.zhihu.com/equation?tex=W_%7Bx%7D%E3%80%81W_%7Bs%7D) 求偏导的公式：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%7BL_%7Bt%7D%7D%7D%7B%5Cpartial%7BW_%7Bx%7D%7D%7D%3D%5Csum_%7Bk%3D0%7D%5E%7Bt%7D%7B%5Cfrac%7B%5Cpartial%7BL_%7Bt%7D%7D%7D%7B%5Cpartial%7BO_%7Bt%7D%7D%7D%5Cfrac%7B%5Cpartial%7BO_%7Bt%7D%7D%7D%7B%5Cpartial%7BS_%7Bt%7D%7D%7D%7D%28%5Cprod_%7Bj%3Dk%2B1%7D%5E%7Bt%7D%7B%5Cfrac%7B%5Cpartial%7BS_%7Bj%7D%7D%7D%7B%5Cpartial%7BS_%7Bj-1%7D%7D%7D%7D%29%5Cfrac%7B%5Cpartial%7BS_%7Bk%7D%7D%7D%7B%5Cpartial%7BW_%7Bx%7D%7D%7D)

任意时刻对![[公式]](https://www.zhihu.com/equation?tex=W_%7Bs%7D) 求偏导的公式同上。

如果加上激活函数， ![[公式]](https://www.zhihu.com/equation?tex=S_%7Bj%7D%3Dtanh%28W_%7Bx%7DX_%7Bj%7D%2BW_%7Bs%7DS_%7Bj-1%7D%2Bb_%7B1%7D%29) ，

则 ![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5E%7Bt%7D%7B%5Cfrac%7B%5Cpartial%7BS_%7Bj%7D%7D%7D%7B%5Cpartial%7BS_%7Bj-1%7D%7D%7D%7D) = ![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5E%7Bt%7D%7Btanh%5E%7B%27%7D%7DW_%7Bs%7D)

激活函数tanh和它的导数图像如下。



![img](https://pic2.zhimg.com/v2-58e93fc298777fb33776f0fe963fedad_r.jpg)

由上图可以看出 ![[公式]](https://www.zhihu.com/equation?tex=tanh%5E%7B%27%7D%5Cleq1) ，对于训练过程大部分情况下tanh的导数是小于1的，因为很少情况下会出现![[公式]](https://www.zhihu.com/equation?tex=W_%7Bx%7DX_%7Bj%7D%2BW_%7Bs%7DS_%7Bj-1%7D%2Bb_%7B1%7D%3D0) ，如果 ![[公式]](https://www.zhihu.com/equation?tex=W_%7Bs%7D) 也是一个大于0小于1的值，则当t很大时 ![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5E%7Bt%7D%7Btanh%5E%7B%27%7D%7DW_%7Bs%7D) ，就会趋近于0，和 ![[公式]](https://www.zhihu.com/equation?tex=0.01%5E%7B50%7D) 趋近与0是一个道理。同理当 ![[公式]](https://www.zhihu.com/equation?tex=W_%7Bs%7D) 很大时 ![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5E%7Bt%7D%7Btanh%5E%7B%27%7D%7DW_%7Bs%7D) 就会趋近于无穷，这就是RNN中梯度消失和爆炸的原因。

至于怎么避免这种现象，让我在看看 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial%7BL_%7Bt%7D%7D%7D%7B%5Cpartial%7BW_%7Bx%7D%7D%7D%3D%5Csum_%7Bk%3D0%7D%5E%7Bt%7D%7B%5Cfrac%7B%5Cpartial%7BL_%7Bt%7D%7D%7D%7B%5Cpartial%7BO_%7Bt%7D%7D%7D%5Cfrac%7B%5Cpartial%7BO_%7Bt%7D%7D%7D%7B%5Cpartial%7BS_%7Bt%7D%7D%7D%7D%28%5Cprod_%7Bj%3Dk%2B1%7D%5E%7Bt%7D%7B%5Cfrac%7B%5Cpartial%7BS_%7Bj%7D%7D%7D%7B%5Cpartial%7BS_%7Bj-1%7D%7D%7D%7D%29%5Cfrac%7B%5Cpartial%7BS_%7Bk%7D%7D%7D%7B%5Cpartial%7BW_%7Bx%7D%7D%7D) 梯度消失和爆炸的根本原因就是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bj%3Dk%2B1%7D%5E%7Bt%7D%7B%5Cfrac%7B%5Cpartial%7BS_%7Bj%7D%7D%7D%7B%5Cpartial%7BS_%7Bj-1%7D%7D%7D%7D) 这一坨，要消除这种情况就需要把这一坨在求偏导的过程中去掉，至于怎么去掉，一种办法就是使 ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cfrac%7B%5Cpartial%7BS_%7Bj%7D%7D%7D%7B%5Cpartial%7BS_%7Bj-1%7D%7D%7D%7D%5Capprox1) 另一种办法就是使 ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cfrac%7B%5Cpartial%7BS_%7Bj%7D%7D%7D%7B%5Cpartial%7BS_%7Bj-1%7D%7D%7D%7D%5Capprox0) 。其实这就是LSTM做的事情，至于细节问题我在[LSTM如何解决梯度消失问题](https://zhuanlan.zhihu.com/p/28749444)这篇文章中给出了介绍。

来源：https://zhuanlan.zhihu.com/p/28687529



# 6. LSTM如何解决梯度消失的问题

> ﻿本篇文章参考于 [RNN梯度消失和爆炸的原因](https://zhuanlan.zhihu.com/p/28687529)、[Towser](https://www.zhihu.com/people/SeptEnds)关于[LSTM如何来避免梯度弥散和梯度爆炸？](https://www.zhihu.com/question/34878706)的问题解答、[Why LSTMs Stop Your Gradients From Vanishing: A View from the Backwards Pass](https://weberna.github.io/blog/2017/11/15/LSTM-Vanishing-Gradients.html)。
> 看本篇文章之前，建议自行学习RNN和LSTM的前向和反向传播过程，学习教程可参考刘建平老师博客[循环神经网络(RNN)模型与前向反向传播算法](https://www.cnblogs.com/pinard/p/6509630.html)、[LSTM模型与前向反向传播算法](https://www.cnblogs.com/pinard/p/6519110.html)。

具体了解LSTM如何解决RNN所带来的梯度消失问题之前，我们需要明白为什么RNN会带来梯度消失问题。

## 6.1. RNN梯度消失原因



![img](https://pic1.zhimg.com/v2-9318154d580e972b8b995d56d16cea4c_r.jpg)

如上图所示，为RNN模型结构，前向传播过程包括，

- **隐藏状态：**![[公式]](https://www.zhihu.com/equation?tex=h%5E%7B%28t%29%7D+%3D+%5Csigma+%28z%5E%7B%28t%29%7D%29+%3D+%5Csigma%28Ux%5E%7B%28t%29%7D+%2B+Wh%5E%7B%28t-1%29%7D+%2B+b%29) ，此处激活函数一般为 ![[公式]](https://www.zhihu.com/equation?tex=tanh) 。
- **模型输出：**![[公式]](https://www.zhihu.com/equation?tex=o%5E%7B%28t%29%7D+%3D+Vh%5E%7B%28t%29%7D+%2B+c)
- **预测输出：**![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D%5E%7B%28t%29%7D+%3D+%5Csigma%28o%5E%7B%28t%29%7D%29) ，此处激活函数一般为softmax。
- **模型损失：**![[公式]](https://www.zhihu.com/equation?tex=L+%3D+%5Csum_%7Bt+%3D+1%7D%5E%7BT%7D+L%5E%7B%28t%29%7D)

RNN反向传播过程中，需要计算 ![[公式]](https://www.zhihu.com/equation?tex=U%2C+V%2C+W) 等参数的梯度，以 ![[公式]](https://www.zhihu.com/equation?tex=W) 的梯度表达式为例，

![[公式]](https://www.zhihu.com/equation?tex=+%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+W%7D+%3D+%5Csum_%7Bt+%3D+1%7D%5E%7BT%7D+%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+y%5E%7B%28T%29%7D%7D+%5Cfrac%7B%5Cpartial+y%5E%7B%28T%29%7D%7D%7B%5Cpartial+o%5E%7B%28T%29%7D%7D+%5Cfrac%7B%5Cpartial+o%5E%7B%28T%29%7D%7D%7B%5Cpartial+h%5E%7B%28T%29%7D%7D+%5Cfrac%7B%5Cpartial+h%5E%7B%28T%29%7D%7D%7B%5Cpartial+h%5E%7B%28t%29%7D%7D+%5Cfrac%7B%5Cpartial+h%5E%7B%28t%29%7D%7D%7B%5Cpartial+W%7D+%5C%5C)

现在需要重点计算 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+h%5E%7B%28T%29%7D%7D%7B%5Cpartial+h%5E%7B%28t%29%7D%7D) 部分，展开得到， ![[公式]](https://www.zhihu.com/equation?tex=+%5Cfrac%7B%5Cpartial+h%5E%7B%28T%29%7D%7D%7B%5Cpartial+h%5E%7B%28t%29%7D%7D+%3D+%5Cfrac%7B%5Cpartial+h%5E%7B%28T%29%7D%7D%7B%5Cpartial+h%5E%7B%28T-1%29%7D%7D+%5Cfrac%7B%5Cpartial+h%5E%7B%28T+-+1%29%7D%7D%7B%5Cpartial+h%5E%7B%28T-2%29%7D%7D+...%5Cfrac%7B%5Cpartial+h%5E%7B%28t%2B1%29%7D%7D%7B%5Cpartial+h%5E%7B%28t%29%7D%7D+%3D+%5Cprod_%7Bk%3Dt+%2B+1%7D%5E%7BT%7D+%5Cfrac%7B%5Cpartial+h%5E%7B%28k%29%7D%7D%7B%5Cpartial+h%5E%7B%28k+-+1%29%7D%7D+%3D+%5Cprod_%7Bk%3Dt%2B1%7D%5E%7BT%7D+tanh%5E%7B%27%7D%28z%5E%7B%28k%29%7D%29+W+%5C%5C)

那么 ![[公式]](https://www.zhihu.com/equation?tex=W) 的梯度表达式也就是，

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+W%7D+%3D+%5Csum_%7Bt+%3D+1%7D%5E%7BT%7D+%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+y%5E%7B%28T%29%7D%7D+%5Cfrac%7B%5Cpartial+y%5E%7B%28T%29%7D%7D%7B%5Cpartial+o%5E%7B%28T%29%7D%7D+%5Cfrac%7B%5Cpartial+o%5E%7B%28T%29%7D%7D%7B%5Cpartial+h%5E%7B%28T%29%7D%7D+%5Cleft%28+%5Cprod_%7Bk%3Dt+%2B+1%7D%5E%7BT%7D+%5Cfrac%7B%5Cpartial+h%5E%7B%28k%29%7D%7D%7B%5Cpartial+h%5E%7B%28k+-+1%29%7D%7D+%5Cright%29+%5Cfrac%7B%5Cpartial+h%5E%7B%28t%29%7D%7D%7B%5Cpartial+W%7D+%5C++%5C%5C+%3D+%5Csum_%7Bt+%3D+1%7D%5E%7BT%7D+%5Cfrac%7B%5Cpartial+L%7D%7B%5Cpartial+y%5E%7B%28T%29%7D%7D+%5Cfrac%7B%5Cpartial+y%5E%7B%28T%29%7D%7D%7B%5Cpartial+o%5E%7B%28T%29%7D%7D+%5Cfrac%7B%5Cpartial+o%5E%7B%28T%29%7D%7D%7B%5Cpartial+h%5E%7B%28T%29%7D%7D+%5Cleft%28+%5Cprod_%7Bk%3Dt%2B1%7D%5E%7BT%7D+tanh%5E%7B%27%7D%28z%5E%7B%28k%29%7D%29+W+%5Cright%29+%5Cfrac%7B%5Cpartial+h%5E%7B%28t%29%7D%7D%7B%5Cpartial+W%7D+%5C++%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=tanh%5E%7B%27%7D%28z%5E%7B%28k%29%7D%29+%3D+diag%281-%28z%5E%7B%28k%29%7D%29%5E2%29+%5Cleq+1) ，随着梯度的传导，如果 ![[公式]](https://www.zhihu.com/equation?tex=W) 的主特征值小于1，梯度便会消失，如果W的特征值大于1，梯度便会爆炸。

需要注意的是，RNN和DNN梯度消失和梯度爆炸含义并不相同。RNN中权重在各时间步内共享，最终的梯度是各个时间步的梯度和。因此，RNN中总的梯度是不会消失的，即使梯度越传越弱，也只是远距离的梯度消失。 **RNN所谓梯度消失的真正含义是，梯度被近距离梯度主导，远距离梯度很小，导致模型难以学到远距离的信息。** 明白了RNN梯度消失的原因之后，我们看LSTM如何解决问题的呢？

## 6.2. LSTM为什么有效？



![img](https://pic1.zhimg.com/v2-39d6bd666a609aa9bd5c09563512f324_r.jpg)



如上图所示，为RNN门控结构，前向传播过程包括，

- **遗忘门输出：**![[公式]](https://www.zhihu.com/equation?tex=f%5E%7B%28t%29%7D+%3D+%5Csigma%28W_fh%5E%7B%28t-1%29%7D+%2B+U_fx%5E%7B%28t%29%7D+%2B+b_f%29)
- **输入门输出：**![[公式]](https://www.zhihu.com/equation?tex=i%5E%7B%28t%29%7D+%3D+%5Csigma%28W_ih%5E%7B%28t-1%29%7D+%2B+U_ix%5E%7B%28t%29%7D+%2B+b_i%29) , ![[公式]](https://www.zhihu.com/equation?tex=a%5E%7B%28t%29%7D+%3D+tanh%28W_ah%5E%7B%28t-1%29%7D+%2B+U_ax%5E%7B%28t%29%7D+%2B+b_a%29)
- **细胞状态：**![[公式]](https://www.zhihu.com/equation?tex=C%5E%7B%28t%29%7D+%3D+C%5E%7B%28t-1%29%7D%5Codot+f%5E%7B%28t%29%7D+%2B+i%5E%7B%28t%29%7D%5Codot+a%5E%7B%28t%29%7D)
- **输出门输出：**![[公式]](https://www.zhihu.com/equation?tex=o%5E%7B%28t%29%7D+%3D+%5Csigma%28W_oh%5E%7B%28t-1%29%7D+%2B+U_ox%5E%7B%28t%29%7D+%2B+b_o%29) , ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7B%28t%29%7D+%3D+o%5E%7B%28t%29%7D%5Codot+tanh%28C%5E%7B%28t%29%7D%29)
- **预测输出：**![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D%5E%7B%28t%29%7D+%3D+%5Csigma%28Vh%5E%7B%28t%29%7D%2Bc%29)

RNN梯度消失的原因是，随着梯度的传导，梯度被近距离梯度主导，模型难以学习到远距离的信息。具体原因也就是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bk%3Dt%2B1%7D%5E%7BT%7D%5Cfrac%7B%5Cpartial+h%5E%7B%28k%29%7D%7D%7B%5Cpartial+h%5E%7B%28k+-+1%29%7D%7D) 部分，在迭代过程中，每一步 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+h%5E%7B%28k%29%7D%7D%7B%5Cpartial+h%5E%7B%28k+-+1%29%7D%7D) **始终在[0,1]之间或者始终大于1。**

而对于LSTM模型而言，针对 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+C%5E%7B%28k%29%7D%7D%7B%5Cpartial+C%5E%7B%28k-1%29%7D%7D) 求得，

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+C%5E%7B%28k%29%7D%7D%7B%5Cpartial+C%5E%7B%28k-1%29%7D%7D+%3D+%5Cfrac%7B%5Cpartial+C%5E%7B%28k%29%7D%7D%7B%5Cpartial+f%5E%7B%28k%29%7D%7D+%5Cfrac%7B%5Cpartial+f%5E%7B%28k%29%7D%7D%7B%5Cpartial+h%5E%7B%28k-1%29%7D%7D+%5Cfrac%7B%5Cpartial+h%5E%7B%28k-1%29%7D%7D%7B%5Cpartial+C%5E%7B%28k-1%29%7D%7D+%2B+%5Cfrac%7B%5Cpartial+C%5E%7B%28k%29%7D%7D%7B%5Cpartial+i%5E%7B%28k%29%7D%7D+%5Cfrac%7B%5Cpartial+i%5E%7B%28k%29%7D%7D%7B%5Cpartial+h%5E%7B%28k-1%29%7D%7D+%5Cfrac%7B%5Cpartial+h%5E%7B%28k-1%29%7D%7D%7B%5Cpartial+C%5E%7B%28k-1%29%7D%7D++%5C%5C+%2B+%5Cfrac%7B%5Cpartial+C%5E%7B%28k%29%7D%7D%7B%5Cpartial+a%5E%7B%28k%29%7D%7D+%5Cfrac%7B%5Cpartial+a%5E%7B%28k%29%7D%7D%7B%5Cpartial+h%5E%7B%28k-1%29%7D%7D+%5Cfrac%7B%5Cpartial+h%5E%7B%28k-1%29%7D%7D%7B%5Cpartial+C%5E%7B%28k-1%29%7D%7D+%2B+%5Cfrac%7B%5Cpartial+C%5E%7B%28k%29%7D%7D%7B%5Cpartial+C%5E%7B%28k-1%29%7D%7D++%5C%5C)

具体计算后得到，

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+C%5E%7B%28k%29%7D%7D%7B%5Cpartial+C%5E%7B%28k-1%29%7D%7D+%3D+C%5E%7B%28k-1%29%7D%5Csigma%5E%7B%27%7D%28%5Ccdot%29W_fo%5E%7B%28k-1%29%7Dtanh%5E%7B%27%7D%28C%5E%7B%28k-1%29%7D%29++%5C%5C+%2B+a%5E%7B%28k%29%7D%5Csigma%5E%7B%27%7D%28%5Ccdot%29W_io%5E%7B%28k-1%29%7Dtanh%5E%7B%27%7D%28C%5E%7B%28k-1%29%7D%29++%5C%5C++%2B+i%5E%7B%28k%29%7Dtanh%5E%7B%27%7D%28%5Ccdot%29W_c%2Ao%5E%7B%28k-1%29%7Dtanh%5E%7B%27%7D%28C%5E%7B%28k-1%29%7D%29++%5C%5C+%2B+f%5E%7B%28t%29%7D++%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=+%5Cprod+_%7Bk%3Dt%2B1%7D%5E%7BT%7D+%5Cfrac%7B%5Cpartial+C%5E%7B%28k%29%7D%7D%7B%5Cpartial+C%5E%7B%28k-1%29%7D%7D+%3D+%28f%5E%7B%28k%29%7Df%5E%7B%28k%2B1%29%7D...f%5E%7B%28T%29%7D%29+%2B+other++%5C%5C)

在LSTM迭代过程中，针对 ![[公式]](https://www.zhihu.com/equation?tex=%5Cprod_%7Bk%3Dt%2B1%7D%5E%7BT%7D+%5Cfrac%7B%5Cpartial+C%5E%7B%28k%29%7D%7D%7B%5Cpartial+C%5E%7B%28k-1%29%7D%7D) 而言，每一步![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+C%5E%7B%28k%29%7D%7D%7B%5Cpartial+C%5E%7B%28k-1%29%7D%7D) **可以自主的选择在[0,1]之间，或者大于1**，因为 ![[公式]](https://www.zhihu.com/equation?tex=f%5E%7B%28k%29%7D) 是可训练学习的。那么整体 ![[公式]](https://www.zhihu.com/equation?tex=%5Cprod+_%7Bk%3Dt%2B1%7D%5E%7BT%7D+%5Cfrac%7B%5Cpartial+C%5E%7B%28k%29%7D%7D%7B%5Cpartial+C%5E%7B%28k-1%29%7D%7D) 也就不会一直减小，远距离梯度不至于完全消失，也就能够解决RNN中存在的梯度消失问题。LSTM虽然能够解决梯度消失问题，但并不能够避免梯度爆炸问题，仍有可能发生梯度爆炸。但是，由于LSTM众多门控结构，和普通RNN相比，LSTM发生梯度爆炸的频率要低很多。梯度爆炸可通过梯度裁剪解决。

LSTM遗忘门值可以选择在[0,1]之间，让LSTM来改善梯度消失的情况。也可以选择接近1，让遗忘门饱和，此时远距离信息梯度不消失。也可以选择接近0，此时模型是故意阻断梯度流，遗忘之前信息。更深刻理解可参考[LSTM如何来避免梯度弥散和梯度爆炸？](https://www.zhihu.com/question/34878706)中回答。

来源：https://zhuanlan.zhihu.com/p/136223550



# 7.时间反向传播

# 8.LSTM和GRU的区别

