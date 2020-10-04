## 1.随机变量

* 定义：如果一个随机试验的结果可以用一个变量来表示，那么该变量叫作<b>随机变量</b>b>

  * 离散行随机变量：可以按照一定次序列出的随机变量，常用字母$\xi$、$\eta$等表示。

  * 连续型随机变量：如果变量可以在某个区间任取一实数，即变量的取值是连续的。

    

  ​                                      表1 离散型随机变量$\xi$的分布列

| $\xi$ | $a_1$ | $a_2$ | $\dots$ | $a_n$ |
| :---: | ----- | ----- | ------- | ----- |
|   P   | $p_1$ | $p_2$ | $\dots$ | $p_n$ |

常见的概率分布有伯努利分布、二项分布、泊松分布和正太分布。



## 2.期望和方差

* 期望定义：假设离散型随机变量$\xi$的分布列如表1所示，则称$a_1p_1+a_2p_2+\ldots+a_np_n$为$\xi$的数学期望，记作$E\xi$。期望反映随机变量取值的平均和集中趋势，具有以下性质：
  * 如果$\eta=a\xi+b$，则$E\eta=aE\xi+b$,a、b为常数
  * $E(\xi_1+\xi_2)=E\xi_1+E\xi_2$
* 方差定义：假设离散型随机变量$\xi$的分布列如表1所示，则称$(a_1-E\xi)^2p_1+(a_2-E\xi)^2p_2+\ldots+(a_n-E\xi)^2p_n$为$\xi$的方差，记作$D\xi$,标准差为$\sqrt{D\xi}$。方差和标准差反应随机变量关于期望的稳定、集中与离散的程度。性质：
  * $D(a\xi+b)=a^2D\xi$
  * 如果$\xi \sim B(n,p)$，则$D\xi=np(1-p)$,B表示二项分布

## 3.伯努利分布

伯努利分布又称两点分布，其概率分布列如下：

| $\xi$ | 1    | 0    |
| ----- | ---- | ---- |
| P     | p    | 1-p  |

设概率质量函数为：
$$
f_X(x)=p^x(1-p)^{1-x}=\begin{equation}
\begin{cases}
p& \text{x=1}\\
1-p& \text{x=0}
\end{cases}
\end{equation}
$$
则随机变量$X$的期望为$p$，方差为$p(1-p)$

```python
probs=np.array([0.6,0.4])
face=[0,1]
plt.bar(face,probs)
plt.title("Bernoulli Distribution",fontsize=12)
plt.ylabel("prob",fontsize=12)
plt.xlabel("x",fontsize=12)
axes=plt.gca()
axes.set_ylim([0,1])
```

![image-20200830101451698](http://qfth8dccq.hn-bkt.clouddn.com/images/image-20200830101451698.png)

## 4.二项分布

二项分布即独立的进行n词伯努利实验，如果概率分布列如下，则称$\xi$服从伯努利分布。伯努利分布是二项式分布中n=1的情形。

| $\xi$ | 0                 | 1                     | 2                     | $\dots$  | n                 |
| ----- | ----------------- | --------------------- | --------------------- | -------- | ----------------- |
| P     | $C_n^0p^0(1-p)^n$ | $C_n^1p^1(1-p)^{n-1}$ | $C_n^2p^2(1-p)^{n-2}$ | $\ldots$ | $C_n^np^n(1-p)^0$ |

当一次实验中有多余两种可能的结果，则二项式分布扩展为多项式分布。

```python
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import seaborn as sns

for prob in range(3,10,3):
  x=np.arange(0,25)
  binom=stats.binom.pmf(x,20,0.1*prob)
  plt.plot(x,binom,"-o",label="p={:f}".format(0.1*prob))
  plt.xlabel('Random Variable', fontsize=12)
  plt.ylabel('Probability', fontsize=12)
  plt.title("Binomial Distribution varying p")
  plt.legend()

```

![image-20200830102025313](http://qfth8dccq.hn-bkt.clouddn.com/images/image-20200830102025313.png)

## 5.泊松分布

如果$\xi$的概率分布列为：
$$
P(X=K)=\frac{\lambda ^k}{k!}e^{-\lambda},(k=0,1,...,n)
$$
则称$\xi$服从泊松分布。其中,$\lambda$表示单位时间或者单位面积内随机事件发生的平均概率，当二项式的n很大而p很小时，泊松分布是二项式分布的近似。泊松分布适合描述在单位时间、单位空间内罕见事件发生次数的分布。

```python
for lambd in range(2, 8, 2):
    n = np.arange(0, 10)
    poisson = stats.poisson.pmf(n, lambd)
    plt.plot(n, poisson, '-o', label="λ = {:f}".format(lambd))
    plt.xlabel('Number of Events', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title("Poisson Distribution varying λ")
    plt.legend()
```

![image-20200830102419883](http://qfth8dccq.hn-bkt.clouddn.com/images/image-20200830102419883.png)

## 6.正态分布

$$
f(x;\eta,\sigma)=ae^{\frac{-(x-\eta)^2}{2\sigma^2}}
$$

如果一个函数形如上式，其中,$a,\eta,\sigma$为实数常数，且$a>0$，则称其为高斯函数。

如果随机变量$x$服从一个位置参数为$\eta$、尺寸参数为$\sigma$的概率分布，并且其概率密度函数为：
$$
\phi(x)=\frac{1}{\sqrt{2\pi\sigma}}e^{\frac{-(x-\eta)^2}{2\sigma^2}}
$$
则随机变量$X$服从正太分布，右称高斯分布。

```python
import numpy as np
import matplotlib.pyplot as plt
import math

u=1 #均值
u01=-2
sig=math.sqrt(0.2) #标准差

x=np.linspace(u-3*sig,u+3*sig,50)
y_sig=np.exp(-(x-u)**2/(2*sig**2))/(math.sqrt(2*math.pi)*sig)
print(x)
print("="*20)
print(y_sig)
plt.plot(x,y_sig,"r-",linewidth=2)
plt.grid(True)
plt.show()
```

![image-20200830100541767](http://qfth8dccq.hn-bkt.clouddn.com/images/image-20200830100541767.png)

## 7.条件概率、联合概率和全概率

* 定义：如果$A$和$B$是两个事件，且$P(B)\neq 0$,那么在给定$B$的条件下,$A$发生的概率为:
  $$
  P(A|B)=\frac{P(A \bigcap B  )}{P(B)}
  $$
  其中$P(A \bigcap B)$是联合概率，表示$A$和$B$同时发生的概率，也可记作$P(A,B)$

* 假设$B_n:1,2,3,\ldots$为有限或者无限个事件，它们两两互斥并且在每次试验中至少有一个发生，则称$B_n$为一完备事件组，且每个集合$B_n$都是一个可测集合，则对任意事件$A$有全概率公式：

$$
P(A)=\sum_nP(A\bigcap B)=\sum _nP(A|B_n)P(B_n)
$$

## 8.先验概率和后验概率

**先验概率（prior probability）：**指根据以往经验和分析。在实验或采样前就可以得到的概率。

**后验概率（posterior probability）：**指某件事已经发生，想要计算这件事发生的原因是由某个因素引起的概率。



## 9.贝叶斯公式

* 定义：设$B_1,B_2,\ldots,B_n$是互不相容的非零概率事件完备系，则对任意非零概率的事件$A$和$k=1,\ldots,n$，有

$$
P(B_k|A)=\frac{P(B_k)P(A|B_k)}{\sum_{j=1}^nP(B_j)P(A|B_j)}
$$

## 10.最大似然估计

在机器学习中，似然函数是一种关于模型中参数的函数。似然函数基于参数的似然性，"似然性"与“概率”的意思很近，但又有所区别。概率用于在已知参数的情况下，预测接下来的观测发生的结果；似然性用于根据一些观测结果，估计给定模型的参数可能值。假设$X$是观测结果序列，它的概率分布$f(x)$依赖参数$\theta$,则似然函数表示为
$$
L(\theta|x)=f_\theta(x)=P_\theta(X=x)
$$
最大似然估计的思想是假设每个观测结果$x$是独立同分布的，通过似然函数$L(\theta|x)$求观测结果$X$发生的概率最大的参数$\theta$，即$argmax_\theta f(X;\theta)$。比如在伯努利分布中,参数$\theta$就是$P$;在泊松分布中，$\theta$代表$\lambda$。

求解最大似然估计的一般步骤如下：

* 写出似然函数
* 对似然函数取对数，得到对数似然函数
* 求对数似然函数的关于参数组的偏导数，并令其为0，得到似然方程组
* 解似然函数，得到参数组的值





## 参考

* [1] <<智能问答与深度学习>>
* [2]https://www.cnblogs.com/Renyi-Fan/p/13282258.html

* [3]https://juejin.im/post/6844904096806223885