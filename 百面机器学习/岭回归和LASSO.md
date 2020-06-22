## **正文**

通过上一篇《[机器学习算法实践-标准与局部加权线性回归](https://zhuanlan.zhihu.com/p/30422174)》中标准线性回归的公式 ![[公式]](https://www.zhihu.com/equation?tex=w+%3D+%28X%5E%7BT%7DX%29%5E%7B-1%7DX%5E%7BT%7Dy) 中可以看出在计算回归系数的时候我们需要计算矩阵 ![[公式]](https://www.zhihu.com/equation?tex=X%5ETX) 的逆，但是如果该矩阵是个奇异矩阵，则无法对其进行求解。那么什么情况下该矩阵会有奇异性呢?

1. X本身存在线性相关关系(多重共线性), 即非满秩矩阵。如果数据的特征中存在两个相关的变量，即使并不是完全线性相关，但是也会造成矩阵求逆的时候造成求解不稳定。
2. 当数据特征比数据量还要多的时候, 即 ![[公式]](https://www.zhihu.com/equation?tex=m+%3C+n) , 这时候矩阵 ![[公式]](https://www.zhihu.com/equation?tex=X) 是一个矮胖型的矩阵，非满秩。

对于上面的两种情况，我们需要对最初的标准线性回归做一定的变化使原先无法求逆的矩阵变得非奇异，使得问题可以稳定求解。我们可以通过缩减的方式来处理这些问题例如岭回归和LASSO.

## **中心化和标准化**

这里先介绍下数据的中心化和标准化，在回归问题和一些机器学习算法中通常要对原始数据进行中心化和标准化处理，也就是需要将数据的均值调整到0，标准差调整为1, 计算过程很简单就是将所有数据减去平均值后再除以标准差:

![[公式]](https://www.zhihu.com/equation?tex=x_i%5E%7B%E2%80%98%7D+%3D+%5Cfrac%7Bx_i+-+%5Cmu%7D%7B%5Csigma%7D)

这样调整后的均值:

![[公式]](https://www.zhihu.com/equation?tex=%5Cmu%5E%7B%E2%80%98%7D+%3D+%5Cfrac%7B%28%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dx_i%29%2Fn+-+%5Cmu%7D%7B%5Csigma%7D+%3D+0)

调整后的标准差:

![[公式]](https://www.zhihu.com/equation?tex=%5Csigma%5E%7B%E2%80%98%7D+%3D+%5Cfrac%7B%28x_i+-+%5Cmu%29%5E2%2Fn%7D%7B%5Csigma%5E2%7D+%3D+%5Cfrac%7B%5Csigma%5E2%7D%7B%5Csigma%5E2%7D+%3D+1)

之所以需要进行中心化其实就是个平移过程，将所有数据的中心平移到原点。而标准化则是使得所有数据的不同特征都有相同的尺度Scale, 这样在使用梯度下降法以及其他方法优化的时候不同特征参数的影响程度就会一致了。

如下图所示，可以看出得到的标准化数据在每个维度上的尺度是一致的(图片来自网络，侵删)

![img](https://pic3.zhimg.com/v2-9d7403075b128631b058f8bdfd7d5d56_r.jpg)

## **岭回归(Ridge Regression)**

标准最小二乘法优化问题:

![[公式]](https://www.zhihu.com/equation?tex=f%28w%29+%3D+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%28y_i+-+x_%7Bi%7D%5E%7BT%7Dw%29%5E2)

也可以通过矩阵表示:

![[公式]](https://www.zhihu.com/equation?tex=f%28w%29+%3D+%28y+-+Xw%29%5E%7BT%7D%28y+-+Xw%29)

得到的回归系数为:

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bw%7D+%3D+%28X%5E%7BT%7DX%29%5E%7B-1%7DX%5E%7BT%7Dy)

这个问题解存在且唯一的条件就是XX列满秩: ![[公式]](https://www.zhihu.com/equation?tex=rank%28X%29+%3D+dim%28X%29) .

即使 ![[公式]](https://www.zhihu.com/equation?tex=X) 列满秩，但是当数据特征中存在共线性，即相关性比较大的时候，会使得标准最小二乘求解不稳定, ![[公式]](https://www.zhihu.com/equation?tex=X%5ETX) 的行列式接近零，计算 ![[公式]](https://www.zhihu.com/equation?tex=X%5ETX) 的时候误差会很大。这个时候我们需要在cost function上添加一个惩罚项 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dw_%7Bi%7D%5E2) ，称为L2正则化。

这个时候的cost function的形式就为:

![[公式]](https://www.zhihu.com/equation?tex=f%28w%29+%3D+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%28y_i+-+x_%7Bi%7D%5E%7BT%7Dw%29%5E2+%2B+%5Clambda%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dw_%7Bi%7D%5E%7B2%7D)

通过加入此惩罚项进行优化后，限制了回归系数wiwi的绝对值，数学上可以证明上式的等价形式如下:

![[公式]](https://www.zhihu.com/equation?tex=f%28w%29+%3D+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%28y_i+-+x_%7Bi%7D%5E%7BT%7Dw%29%5E2+)

![[公式]](https://www.zhihu.com/equation?tex=s.t.+%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dw_%7Bj%7D%5E2+%5Cle+t)

其中t为某个阈值。

将岭回归系数用矩阵的形式表示:

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bw%7D+%3D+%28X%5E%7BT%7DX+%2B+%5Clambda+I%29%5E%7B-1%7DX%5E%7BT%7Dy)

可以看到，就是通过将 ![[公式]](https://www.zhihu.com/equation?tex=X%5ETX) 加上一个单位矩阵是的矩阵变成非奇异矩阵并可以进行求逆运算。

## **岭回归的几何意义**

以两个变量为例, 残差平方和可以表示为 ![[公式]](https://www.zhihu.com/equation?tex=w_1%2C+w_2) 的一个二次函数，是一个在三维空间中的抛物面，可以用等值线来表示。而限制条件 ![[公式]](https://www.zhihu.com/equation?tex=w_1%5E2+%2B+w_2%5E2+%3C+t) ， 相当于在二维平面的一个圆。这个时候等值线与圆相切的点便是在约束条件下的最优点，如下图所示，

![img](https://pic3.zhimg.com/80/v2-a13db3079fd191c00150eec088c1530e_720w.jpg)

## **岭回归的一些性质**

1. 当岭参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda+%3D+0) 时，得到的解是最小二乘解
2. 当岭参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) 趋向更大时，岭回归系数 ![[公式]](https://www.zhihu.com/equation?tex=w_i) 趋向于0，约束项 ![[公式]](https://www.zhihu.com/equation?tex=t) 很小

## **岭迹图**

可以知道求得的岭系数 ![[公式]](https://www.zhihu.com/equation?tex=w_i) 是岭参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) 的函数，不同的 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) 得到不同的岭参数 ![[公式]](https://www.zhihu.com/equation?tex=w_i) , 因此我们可以增大 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) 的值来得到岭回归系数的变化，以及岭参数的变化轨迹图(岭迹图), 不存在奇异性时，岭迹图应稳定的逐渐趋向于0。

通过岭迹图我们可以:

1. 观察较佳的 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) 取值
2. 观察变量是否有多重共线性



## **LASSO**

岭回归限定了所有回归系数的平方和不大于 ![[公式]](https://www.zhihu.com/equation?tex=t) , 在使用普通最小二乘法回归的时候当两个变量具有相关性的时候，可能会使得其中一个系数是个很大正数，另一个系数是很大的负数。通过岭回归的 ![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+w_i+%5Cle+t) 的限制，可以避免这个问题。

LASSO(The Least Absolute Shrinkage and Selection Operator)是另一种缩减方法，将回归系数收缩在一定的区域内。LASSO的主要思想是构造一个一阶惩罚函数获得一个精炼的模型, 通过最终确定一些变量的系数为0进行特征筛选。

LASSO的惩罚项为:

![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5E%7Bn%7D+%5Cvert+w_i+%5Cvert+%5Cle+t)

与岭回归的不同在于，此约束条件使用了绝对值的一阶惩罚函数代替了平方和的二阶函数。虽然只是形式稍有不同，但是得到的结果却又很大差别。在LASSO中，当λλ很小的时候，一些系数会随着变为0而岭回归却很难使得某个系数**恰好**缩减为0. 我们可以通过几何解释看到LASSO与岭回归之间的不同。

## **LASSO的几何解释**

同样以两个变量为例，标准线性回归的cost function还是可以用二维平面的等值线表示，而约束条件则与岭回归的圆不同，LASSO的约束条件可以用方形表示，如下图:

![img](https://pic2.zhimg.com/80/v2-e16c01b5088ed0e6dc37833002e7b769_720w.jpg)

相比圆，方形的顶点更容易与抛物面相交，顶点就意味着对应的很多系数为0，而岭回归中的圆上的任意一点都很容易与抛物面相交很难得到正好等于0的系数。这也就意味着，lasso起到了很好的筛选变量的作用。

## **LASSO回归系数的计算**

虽然惩罚函数只是做了细微的变化，但是相比岭回归可以直接通过矩阵运算得到回归系数相比，LASSO的计算变得相对复杂。由于惩罚项中含有绝对值，此函数的导数是连续不光滑的，所以无法进行求导并使用梯度下降优化。本部分使用坐标下降发对LASSO回归系数进行计算。

坐标下降法是每次选择一个维度的参数进行一维优化，然后不断的迭代对多个维度进行更新直到函数收敛。SVM对偶问题的优化算法SMO也是类似的原理，这部分的详细介绍我在之前的一篇博客中进行了整理，参考《[机器学习算法实践-SVM中的SMO算法](https://link.zhihu.com/?target=http%3A//pytlab.github.io/2017/09/01/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95%E5%AE%9E%E8%B7%B5-SVM%E4%B8%AD%E7%9A%84SMO%E7%AE%97%E6%B3%95/)》。

下面我们分别对LASSO的cost function的两部分求解:

1) RSS部分

![[公式]](https://www.zhihu.com/equation?tex=RSS%28w%29+%3D+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28y_i+-+%5Csum_%7Bj%3D1%7D%5E%7Bn%7Dx_%7Bij%7Dw_j%29%5E2)

求导:

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+RSS%28w%29%7D%7B%5Cpartial+w_k%7D+%3D+-2%5Csum_%7Bi%3D1%7D%5E%7Bm%7Dx_%7Bik%7D%28y_i+-+%5Csum_%7Bj%3D1%7D%5E%7Bn%7Dx_%7Bij%7Dw_j%29)

![[公式]](https://www.zhihu.com/equation?tex=%3D+-2%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%28x_%7Bik%7Dy_i+-+x_%7Bik%7D%5Csum_%7Bj%3D1%2C+j+%5Cne+k%7D%5E%7Bn%7Dx_%7Bij%7Dw_j+-+x_%7Bik%7D%5E2w_k%29)

![[公式]](https://www.zhihu.com/equation?tex=%3D+-2%5Csum_%7Bi%3D1%7D%5E%7Bm%7Dx_%7Bik%7D%28y_i+-+%5Csum_%7Bj%3D1%2C+j+%5Cne+k%7D%5E%7Bn%7Dx_%7Bij%7Dw_%7Bj%7D%29+%2B+2w_k%5Csum_%7Bi%3D1%7D%5E%7Bm%7Dx_%7Bik%7D%5E2)

令 ![[公式]](https://www.zhihu.com/equation?tex=p_k+%3D+%5Csum_%7Bi%3D1%7D%5E%7Bm%7Dx_%7Bik%7D%28y_i+-+%5Csum_%7Bj%3D1%2C+j+%5Cne+k%7D%5E%7Bn%7Dx_%7Bij%7Dw_%7Bj%7D%29) ， ![[公式]](https://www.zhihu.com/equation?tex=z_k+%3D+%5Csum_%7Bi%3D1%7D%7Bm%7Dx_%7Bik%7D%5E2) 得到:

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+RSS%28w%29%7D%7B%5Cpartial+w_j%7D+%3D+-2p_k+%2B+2z_kw_k) ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+RSS%28w%29%7D%7B%5Cpartial+w_j%7D+%3D+-2p_k+%2B+2z_kw_k)

2）正则项

关于惩罚项的求导我们需要使用subgradient，可以参考[LASSO（least absolute shrinkage and selection operator） 回归中 如何用梯度下降法求解？](https://www.zhihu.com/question/22332436/answer/21068494)

![[公式]](https://www.zhihu.com/equation?tex=%5Clambda+%5Cfrac%7B%5Cpartial+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cvert+w_j+%5Cvert%7D%7B%5Cpartial+w_k%7D+%3D+%5Cbegin%7Bcases%7D+-%5Clambda+%26+w_k+%3C+0+%5C%5C+%5B-%5Clambda%2C+%5Clambda%5D+%26+w_k+%3D+0+%5C%5C+%5Clambda+%26+w_k+%3E+0+%5Cend%7Bcases%7D)

这样整体的偏导数:

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%28w%29%7D%7B%5Cpartial+w_k%7D+%3D+2z_kw_k+-+2p_k+%2B+%5Cbegin%7Bcases%7D+-%5Clambda+%26+w_k+%3C+0+%5C%5C+%5B-%5Clambda%2C+%5Clambda%5D+%26+w_k+%3D+0+%5C%5C+%5Clambda+%26+w_k+%3E+0+%5Cend%7Bcases%7D)

![[公式]](https://www.zhihu.com/equation?tex=%3D+%5Cbegin%7Bcases%7D+2z_kw_k+-+2p_k+-+%5Clambda+%26+w_k+%3C+0+%5C%5C+%5B-2p_k+-+%5Clambda%2C+-2p_k+%2B+%5Clambda%5D+%26+w_j+%3D+0+%5C%5C+2z_kw_k+-+2p_k+%2B+%5Clambda+%26+w_k+%3E+0+%5Cend%7Bcases%7D)

令 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%28w%29%7D%7B%5Cpartial+w_k%7D+%3D+0) 得到

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bw_k%7D+%3D+%5Cbegin%7Bcases%7D+%28p_k+%2B+%5Clambda%2F2%29%2Fz_k+%26+p_k+%3C+-%5Clambda%2F2+%5C%5C+0+%26+-%5Clambda%2F2+%5Cle+p_k+%5Cle+%5Clambda%2F2+%5C%5C+%28p_k+-+%5Clambda%2F2%29%2Fz_k+%26+p_k+%3E+%5Clambda%2F2+%5Cend%7Bcases%7D)

通过上面的公式我们便可以每次选取一维进行优化并不断跌打得到最优回归系数。





## **逐步向前回归**

LASSO计算复杂度相对较高，本部分稍微介绍一下逐步向前回归，他属于一种贪心算法，给定初始系数向量，然后不断迭代遍历每个系数，增加或减小一个很小的数，看看代价函数是否变小，如果变小就保留，如果变大就舍弃，然后不断迭代直到回归系数达到稳定。



逐步回归算法的主要有点在于他可以帮助人们理解现有的模型并作出改进。当构建了一个模型后，可以运行逐步回归算法找出重要的特征，即使停止那些不重要特征的收集。

## **总结**

![img](https://pic2.zhimg.com/v2-74227950ad507eb0c2418b2d272e94c5_r.jpg)

本文介绍了两种回归中的缩减方法，岭回归和LASSO。两种回归均是在标准线性回归的基础上加上正则项来减小模型的方差。这里其实便涉及到了权衡偏差(Bias)和方差(Variance)的问题。方差针对的是模型之间的差异，即不同的训练数据得到模型的区别越大说明模型的方差越大。而偏差指的是模型预测值与样本数据之间的差异。所以为了在过拟合和欠拟合之前进行权衡，我们需要确定适当的模型复杂度来使得总误差最小。

## **参考**

- 《Machine Learning in Action》
- [机器学习中的Bias(偏差)，Error(误差)，和Variance(方差)有什么区别和联系？](https://www.zhihu.com/question/27068705)
- [Lasso回归的坐标下降法推导](https://link.zhihu.com/?target=http%3A//blog.csdn.net/u012151283/article/details/77487729)
- [数据什么时候需要做中心化和标准化处理？](https://www.zhihu.com/question/37069477)

来源：https://zhuanlan.zhihu.com/p/30535220