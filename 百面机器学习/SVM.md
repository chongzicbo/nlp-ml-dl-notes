# 1.SVM简介

SVM是一种二类分类模型。基本思想是在特征空间中寻找间隔最大的分类超平面使数据得到高效的二分类。具体有三种情况：

* 当训练样 本线性可分时，通过硬间隔最大化，学习一个线性分类器，即线性可分支持向量机。
* 当训练数据近似线性可分时，引入松弛变量，通过软间隔最大化，学习一个线性分类器，即线性支撑向量机。
* 当训练数据线性不可分时，通过使用核技巧及软间隔最大化，学习非线性支持向量机。

# 2.SVM为什么采用间隔最大化（与感知机的区别）

当训练数据线性可分时，存在无穷个分离超平面可以将两类数据正确分开。**感知机利用误分类最小策略**，求得分离超平面，不过此时的**解有无穷多个**。线性可分支持向量机利用间隔最大化求得最优分离超平面，这时，解是唯一的。另一方面，此时的分隔超平面所产生的分类结果是**最鲁棒**的，对未知实例的**泛化能力最强**。

# 3. SVM的目标（硬间隔）

两个目标：

* 间隔最大化
  $$
  \min_{w,b}\frac{1}{2}||w||
  $$
  

* 使样本正确分类
  $$
  y_i(w^Tx_i+b)\geq1,i=1,2,\dots,m
  $$
  其中$w$是超平面参数，目标1是从点到面的距离公式简化而来，目标2相当于感知机，只是把大于等于0缩放成了大于等于1，为了方便推导。有了两个目标，写在一起，就变成了SVM的终极目标：
  $$
  \min _{w,b}\frac{1}{2}||w||^2 \\
  s.t. y_i(w^Tx_i+b)\geq1,\forall i
  $$
  

# 4. 求解目标(硬间隔)

从目标公式可知是一个有约束条件的最优化问题，用拉格朗日函数解决：
$$
\min_{w,b}\max_\alpha L(w,b,\alpha)=\frac{1}{2}||w||^2+\sum_{i=1}^m \alpha _i(1-y_i(w^Tx_i+b)) \\
s.t. \alpha _i \geq0,\forall i
$$
在满足Slater定理的时候，且过程满足KKT条件时，原问题可转为对偶问题：
$$
\max_\alpha\min_{w,b} L(w,b,\alpha)=\frac{1}{2}||w||^2+\sum_{i=1}^m \alpha _i(1-y_i(w^Tx_i+b)) \\
s.t. \alpha _i \geq0,\forall i
$$
先求其内部最小值，对$w$和$b$求偏导并令其等于0可得：
$$
w=\sum _{i=1}^m \alpha_iy_ix_i \\
\sum_{i=1}^m \alpha _i y_i
$$
将其带入上式中可得：
$$
\max _\alpha L(w,b,\alpha)=\sum_{i=1}^m\alpha_i-\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy_iy_jx_i^Tx_j\\
s.t. \sum_{i=1}^m\alpha_iy_i=0(\alpha_i \geq0,i=1,2,\ldots,m)
$$
此时需要求解$\alpha$，l利用SMO算法：

SMO的算法的基本思路是每次选择两个变量$\alpha _i$和$\alpha _j$，选取的两个变量所对应的样本之间间隔要尽可能大，因为这样更新会带给目标函数值更大的变化。SMO算法之所以高效，是因为仅优化两个参数的过程实际仅有一个约束条件，其中一个可由另一个表示，这样的二次规划问题具有闭式解。

# 5. 软间隔

不管直接在原特征空间，还是在映射的高维空间，我们都假设样本是线性可分的。虽然理论上我们总能找到一个高维映射使数据线性可分，但在实际任务中，寻找一个合适的核函数核很困难。此外，由于数据通常有噪声存在，一味追求数据线性可分可能会使模型陷入过拟合，因此，我们放宽对样本的要求，允许少量样本分类错误。这样的想法就意味着对目标函数的改变，之前推导的目标函数里不允许任何错误，并且让间隔最大，现在给之前的目标函数加上一个误差，就相当于允许原先的目标出错，引入松弛变量$\xi _i \geq0$,公式变为：
$$
\min _{w,b,\xi}\frac{1}{2}||w||^2+\sum_{i=1}^m \xi_i
$$
那么这个松弛变量怎么计算呢，最开始试图用0，1损失去计算，但0，1损失函数并不连续，求最值时求导的时候不好求，所以引入合页损失（hinge loss）：
$$
l_{hinge}(z)=\max(0,1-z)
$$
函数图长这样：

![img](https://mmbiz.qpic.cn/mmbiz_png/DHibuUfpZvQeuNqtzSBzsAQFKc4YF8vawIxibw1tgLydx8u6LReOZTtDcico0tM6RL5gic2Xzic8wwnvgNvufAqK13A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

理解起来就是，原先制约条件是保证所有样本分类正确，$y_i(w^Tx_i+b)\geq1,\forall i$ ，现在出现错误的时候，一定是这个式子不被满足了，即 $y_i(w^Tx_i+b)<1,\forall i_{错误}$，衡量一下错了多少呢？因为左边一定小于1，那就跟1比较，因为1是边界，所以用1减去$y_i(w^Tx_i+b)$ 来衡量错误了多少，所以目标变为（正确分类的话损失为0，错误的话付出代价）：
$$
\min _{w,b}\frac{1}{2}||w||^2+\sum_{i=1}^m \max(0,1-y_i(w^Tx_i+b))
$$
但这个代价需要一个控制的因子，引入$C >0$,惩罚参数，即：
$$
\min _{w,b}\frac{1}{2}||w||^2+C\sum_{i=1}^m \max(0,1-y_i(w^Tx_i+b))
$$
可以想象，C越大说明把错误放的越大，说明对错误的容忍度就小，反之亦然。当C无穷大时，就变成一点错误都不能容忍，即变成硬间隔。实际应用时我们要合理选取C，C越小越容易欠拟合，C越大越容易过拟合。

所以软间隔的目标函数为：
$$
\min_{w,b\xi}\frac{1}{2}||w||^2+C\sum_{i=1}^n \xi_i \\
s.t. y_i(x_i^Tw+b) \geq 1-\xi_i \\
\xi_i \geq0,i=1,2,\dots,n
$$
其中：
$$
\xi_i=max(0,1-y_i(w^Tx_i+b))
$$

# 6.软间隔求解

与硬间隔类似：

上式的拉格朗日函数为：
$$
\min _{w,b,\xi}\max_{\alpha,\beta}L(w,b,\alpha,\xi,\beta)=\frac{1}{2}||w||^2+C\sum_{i=1}^n\xi_i+\sum_{i=1}^m\alpha_i(1-y_i(w^Tx_i+b)-\xi_i)-\sum_{i=1}^n\beta_i\xi_i \\
s.t.\alpha_i \geq0 且\beta_i \geq0,\forall_i
$$
在满足Slater定理的时候，且过程满足KKT条件的时候，原问题转换成对偶问题：
$$
\max_{\alpha,\beta}\min _{w,b,\xi}L(w,b,\alpha,\xi,\beta)=\frac{1}{2}||w||^2+C\sum_{i=1}^n\xi_i+\sum_{i=1}^m\alpha_i(1-y_i(w^Tx_i+b)-\xi_i)-\sum_{i=1}^n\beta_i\xi_i \\
s.t.\alpha_i \geq0 且\beta_i \geq0,\forall_i
$$
先求内部最小值，对$w,b$和$\xi$求偏导并令其等于0可得：
$$
w=\sum_{i=1}^m \alpha_iy_ix_i \\
0=\sum_{i=1}^m\alpha_iy_i \\
C=\alpha_i+\beta_i
$$
将其代入到上式中去可得到，注意$\beta$ 被消掉了：
$$
\max_{\alpha,\beta}L(w,b,\alpha,\xi,\beta)=\sum_{i=1}^m\alpha_i-\frac{1}{2}\sum_{i=1}^{m}\sum_{j=1}^m\alpha_i\alpha_jy_iy_jx_i^Tx_j \\
s.t.\sum_{i=1}^m\alpha_iy_i=0 \quad (0 \leq\alpha_i\leq C,i=1,2,\dots,m)
$$
此时需求解$\alpha$，同样利用SMO（序列最小最优化）算法。



# 7.和函数

为什么要引入核函数：

> 当样本在原始空间线性不可分时，可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分。而引入这样的映射后，所要求解的对偶问题的求解中，无需求解真正的映射函数，而只需要知道其核函数。核函数的定义：K(x,y)=<ϕ(x),ϕ(y)>，即在特征空间的内积等于它们在原始样本空间中通过核函数 K 计算的结果。一方面数据变成了高维空间中线性可分的数据，另一方面不需要求解具体的映射函数，只需要给定具体的核函数即可，这样使得求解的难度大大降低。

用自己的话说就是，在SVM不论是硬间隔还是软间隔在计算过程中，都有X转置点积X，若X的维度低一点还好算，但当我们想把X从低维映射到高维的时候（让数据变得线性可分时），这一步计算很困难，等于说在计算时，需要先计算把X映射到高维的的ϕ(x)，再计算ϕ(x1)和ϕ(x2)的点积，这一步计算起来开销很大，难度也很大，此时引入核函数，这两步的计算便成了一步计算，即只需把两个x带入核函数，计算核函数，举个列子一目了然（图片来自：从零推导支持向量机）：![img](https://mmbiz.qpic.cn/mmbiz_png/DHibuUfpZvQeuNqtzSBzsAQFKc4YF8vawT8Ndeo8l4QIu6OTnOINIJU0Eiac3OmljT8pVdbficoKIjnwltWibaGzCA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

个人对核函数的理解：核函数就是一个函数，接收两个变量，这两个变量是在低维空间中的变量，而核函数求的值等于将两个低维空间中的向量映射到高维空间后的内积。



# 8.如何确定一个函数是核函数

验证正定核啥的，咱也不太懂，给出：

> 设 $\chi \subset R^n,K(x,z)$ 是定义在$\chi \times \chi$ 上的对称函数，如果对任意的 $x_i \in \chi,i=1,2,...,m,K(x,z)$ 对应的Gram矩阵 是半正定矩阵$K=[K(x_i,x_j)]_{m \times m}$，则$K(x,z)$ 是正定核

所以不懂，就用人家确定好的常见核函数及其优缺点：![img](https://mmbiz.qpic.cn/mmbiz_png/DHibuUfpZvQeuNqtzSBzsAQFKc4YF8vaw5GzT5n3KmMv0Hn6s6V8iaGcZ2Fib6004KlacqsRtTnyib3wusaBib56GJg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

# 9.如何选择核函数

- 当特征维数 超过样本数 时 (文本分类问题通常是这种情况), 使用线性核;
- 当特征维数 比较小. 样本数 中等时, 使用RBF核;
- 当特征维数 比较小. 样本数 特别大时, 支持向量机性能通常不如深度神经网络



# 10.关于支持向量的问题

#### 1. 先说硬间隔：

先看KKT条件

- 主问题可行：$1-y_i(w^Tx_i+b)\leq0$
- 对偶问题可行：$\alpha_i \geq0$
- 互补松弛：$\alpha_i(1-y_i(w^Tx_i+b))=0$

支持向量，对偶变量 $\alpha_i >0$对应的样本；

- 线性支持向量机中, 支持向量是距离划分超平面最近的样本, 落在最大间隔边界上。

> 证明：由线性支持向量机的KKT 条件可知$\alpha_i(1-y_i(w^Tx_i+b))=0$。当 $\alpha_i >0$时，$1-y_i(w^Tx_i+b)=0$，即$y_i(w^Tx_i+b)=1$

* 支持向量机的参数 (w; b) 仅由支持向量决定, 与其他样本无关。

![img](https://mmbiz.qpic.cn/mmbiz_png/DHibuUfpZvQeuNqtzSBzsAQFKc4YF8vawdZ91OicqNLIt3gjqbNDuTtVr6Qrdn6CAoSMgcVlYWeDE5mfZ9ibGH7Ag/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

#### 2. 再说软间隔：

先看kkt条件：

- 主问题可行：$1-\xi_i-y_i(w^T\phi(x_i)+b)\leq0,-\xi_i \leq 0$
- 对偶问题可行：$\alpha _i \geq 0,\beta_i \geq0$
- 互补松弛：$\alpha _i(1-\xi_i-y_i(w^T\phi(x_i)+b))=0,\beta_i\xi_i=0$

经过SMO后，求得$\hat \alpha,0<\hat \alpha _j <C$。

对于任意样本$(X_i,y_i)$

- 若 $\alpha _i =0$，此样本点不是支持向量，该样本对模型没有任何的作用
- 若 $\alpha _i >0$，此样本是一个支持向量（同硬间隔）

若满足$\alpha _i >0$ ，进一步地，