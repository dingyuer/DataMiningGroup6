### T-SNE

是一种降维方法，**t-SNE** (t分布随机邻居嵌入)，可将高维数据降到2~3维进行可视化，PCA是1933年开发的，T-SNE是2008年开发的.

##### 算法步骤：

* 找出高维空间中相邻点之间的**成对相似性**
* 根据高维空间中点的成对相似性，将高维空间中的每个点映射到低维映射
* 使用基于Kullback-Leibler散度(KL散度)的梯度下降找到最小化条件概率分布之间的不匹配的低维数据表示
* 使用Student-t分布计算低维空间中两点之间的相似度

一个简单实现：

```python
import matplotlib.pyplot as plt
import numpy as np

tsne = TSNE(perplexity=2, n_components=2)
x = [[1, 2, 2], [2, 2, 2], [3, 3, 3]]
y = [1, 0, 2]  # y是x对应的标签
x = np.array(x)
y = np.array(y)
x_tsne = tsne.fit_transform(x)
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y)
plt.show()
```

运行图示：

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240423214027825.png" alt="image-20240423214027825" style="zoom: 50%;" />

#### NMI 指标

1、定义：NMI(Normalized Mutual Information) 归一化互信息，常用在聚类种，度量两个聚类结果的相近程度

2、公式：$\displaystyle \texttt{NMI}(Y,C)=\frac{2\times I(Y;C)}{H(Y)+H(C)}$

其中：$Y$代表真实的数据类别，$C$代表聚类的结果，$H(\cdot)$是交叉熵,$H(X)=-\sum\limits_{i=1}^{|X|}\Pr(i)\log\Pr(i)$,

$I(Y;C)$是互信息，$I(Y;C)=H(Y)-H(Y|C)$

3、理解：如图,图像块为真实标签，圆圈为聚类结果

<img src="C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20240423212409975.png" alt="image-20240423212409975" style="zoom: 33%;" />

1. 计算$Y$的交叉熵：

   $P(Y=1)=\frac{1}{4}, P(Y=2)=\frac{1}{4}, P(Y=3)=\frac{1}{2}$

2. 计算$C$的交叉熵：

   $P(C=1)=\frac{1}{2}, P(C=2)=\frac{1}{2}$

3. 计算$Y$与$C$的互信息：

   $P(Y=1|C=1)=\frac{3}{10}, P(Y=2|C=1)=\frac{3}{10}, P(Y=3|C=1)=\frac{4}{10}$

   所以$H(Y|C=1)=-P(C=1)\sum\limits_{y\in \{1,2,3\}}P(Y=y|C=1)\log P(Y=y|C=1)=0.7855$

   同理$H(Y|C=2)=0.5784$

   故$I(Y;C)=H(Y)-H(Y|C)=1.5-(0.7855+0.5784)=0.1361$

4. 最终结果：

   $\texttt{NMI}(Y,C)=\frac{2\times I(Y;C)}{H(Y)+H(C)}=0.1089$

示例代码：

```python
from sklearn.metrics.cluster import normalized_mutual_info_score
labels_true = [0, 0, 1, 1]
labels_pred = [0, 0, 1, 1]

score = normalized_mutual_info_score(labels_true, labels_pred)
print(score) # 1.0

labels_true = [0, 0, 0, 0]
labels_pred = [0, 1, 2, 3]

score = normalized_mutual_info_score(labels_true, labels_pred)
print(score) # 0.0
```



#### ARI 指标

1、定义：ARI（Adjusted Rand Index）调整兰德指数，一种常用的聚类外部指标。

2、公式：$\displaystyle\texttt{ARI}=\frac{\sum_{ij}\binom{n_{ij}}{2}-[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}]/\binom{n}{2}}{\frac{1}{2}[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}]-[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}]/\binom{n}{2}}$

###### 注：这个公式打起来累死了

3、理解：首先看兰德系数，设$U=\{u_1,u_2...u_{k_1}\}$是真实标签$V=\{v_1,v_2...v_{k_2}\}$是聚类所得结果，则兰德系数$\displaystyle RI=\frac{a+d}{a+b+c+d}$，其中

- $a$：在$U$中为同一类且在$V$中也为同一类的数据点对数
- $b$：在$U$中为同一类但在$V$中却为不同类的数据点对数
- $c$：在$U$中为不同类但在$V$中却为同一类的数据点对数
- $d$：在$U$中为不同类且在$V$中也为不同类的数据点对数

其存在的问题：对于随机聚类其值不接近于$0$,因此有了调整兰德系数，其可以被表示为：

$\displaystyle\texttt{ARI}=\frac{RI-E(RI)}{max(RI)-E(RI)}$

示例代码：

```python
from sklearn import metrics
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]

# 基本用法
score = metrics.adjusted_rand_score(labels_true, labels_pred)
print(score)  # 0.24242424242424246

# 与标签名无关
labels_pred = [1, 1, 0, 0, 3, 3]
score = metrics.adjusted_rand_score(labels_true, labels_pred)
print(score)  # 0.24242424242424246

# 具有对称性
score = metrics.adjusted_rand_score(labels_pred, labels_true)
print(score)  # 0.24242424242424246

# 接近 1 最好
labels_pred = labels_true[:]
score = metrics.adjusted_rand_score(labels_true, labels_pred)
print(score)  # 1.0

# 独立标签结果为负或者接近 0
labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
score = metrics.adjusted_rand_score(labels_true, labels_pred)
print(score)  # -0.12903225806451613
```





