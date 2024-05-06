# IDK

$$IK$$: In 2018, a data dependent kernel called Isolation Kernel or IK is  first introduced as an alternative to data independent kernels such as  Gaussian and Laplacian kernels. It has a unique characteristic:  two  points, as measured by Isolation Kernel derived with a dataset in a  sparse region, are more similar than the same two points, as measured by Isolation Kernel derived with a dataset in a dense region. 

$$IDK$$: In 2020, Isolation Distributional Kernel or IDK is introduced to measure the similarity of two distributions, based on the framework of  kernel mean embedding. 

***

### 使用IDK来对两个时间序列做相似性度量，即是将两个时间序列看成是两个分布。

1. 什么是核函数

   $\kappa(x, y)=\phi^T(x)\phi(y)$，$\phi$表示将一个向量映射到某个特征空间，这表明了两个向量在特征空间的内积可以由核函数在原始样本空间中通过核函数计算。有这样的函数就不必去计算高维甚至$\infty$维的特征空间中的内积。

2. KME是什么

   $K(P_X,P_Y)=\frac{1}{|X||Y|}\sum_{x\in X}\sum_{y\in Y}\kappa(x,y)\approx\langle\varphi(P_X),\varphi(P_Y)\rangle$，这将point-to-point kernel变成了distributional kernel，其中核函数是高斯核。

3. IDK呢

   将上述KME中的高斯核替换为IK即可。

4. IK

   实现方式

   Isolation Forest is one of the most effective and efficient anomaly  detectors created thus far. Since its introduction, it has been used  widely in academia and industries. Its limitations, due to the use of  tree structures, have been studied by different researchers. One  improvement beyond tree structures is iNNE which employs  hyperspheres as the isolation mechanism.
   
- Isolation forest
   - iNNE

   
   
   