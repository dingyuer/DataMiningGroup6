import numpy as np
import similaritymeasures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def InitializeCentroids(points, k):
    """
    KMeans聚类算法初始化: 随机选择k个点作为初始的聚类中心
    :param points: 样本集
    :param k: 聚类簇数
    :return: 随机选择的k个聚类中心
    """
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def ClosestCentroid(points, centroids):
    """
    计算每个样本与聚类中心的距离（相似度），将其归入最近（最相近）的簇
    :param points: 样本集
    :param centroids: 聚类中心
    :return: 样本所属聚类的簇
    """
    #Dist = (points[:, np.newaxis, :] - centroids[np.newaxis, :, :]).sum(axis=2)
    #return np.argmin(Dist, axis=1)
    
    # 初始化一个数组来保存每个点到各个中心的MSE距离
    distances = np.zeros((len(points), len(centroids)))
    # 计算每个点与每个中心的距离
    for i, point in enumerate(points):
        for j, centroid in enumerate(centroids):
            distances[i, j] = similaritymeasures.mse(point, centroid)
    
    # 对于每个点，找到距离最小（相似度最大）的中心索引
    return np.argmin(distances, axis=1)

def UpdateCentroids(points, closestCentroid, centroids):
    """
    对每个簇计算所有点的均值作为新的聚类中心
    :param points: 样本集
    :param closestCentroid: 每个点所属簇的索引
    :param centroids: 上一轮迭代的聚类中心
    :return: 新的聚类中心
    """
    newCentroids = []
    for k in range(centroids.shape[0]):
        # 过滤出属于当前簇k的点
        points_k = points[closestCentroid == k]
        # 检查是否为空簇
        if len(points_k) > 0:
            newCentroids.append(points_k.mean(axis=0))
        else:
            # 如果为空簇，随机选择一个新的中心
            newCentroids.append(points[np.random.randint(0, len(points))])
    return np.array(newCentroids)

def KMeans(points, k=3, maxIters=10):
    """
    KMeans聚类算法实现
    :param points: 样本集
    :param k: 聚类簇数
    :param maxIters: 最大迭代次数
    :return: 聚类后的簇划分
    """
    centroids = InitializeCentroids(points=points, k=k)
    for i in range(maxIters):
        closestCentroid = ClosestCentroid(points=points, centroids=centroids)
        newCentroids = UpdateCentroids(points=points, closestCentroid=closestCentroid, centroids=centroids)
        if (newCentroids == centroids).all():    # 聚类中心不再发生改变，停止迭代
            break
        centroids = newCentroids
    return centroids, closestCentroid, points

