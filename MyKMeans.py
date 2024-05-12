import numpy as np
import similaritymeasures
from hausdorff import hausdorff_distance
from scipy.stats import wasserstein_distance
from scipy import spatial
from enum import Enum

class choose(Enum):
    L1 = 1
    L2 = 2
    Cos = 3
    DTW = 4
    GK = 5
    LK = 6
    IDK = 7
    GDK = 8
    IK = 9
    FD = 10
    HD = 11
    WD = 12
    
C = choose.WD

def gdk(S, T):
    length1 = len(S)
    length2 = len(T)
    score = 0.0
    for x in S:
        for y in T:
            score += gaussian_kernel(x, y, 1.0)
    return score / (length1 * length2)

def gaussian_kernel(x, y, sigma=1.0):
    # 计算样本之间的欧氏距离的平方
    distance_squared = np.sum((x - y) ** 2)
    # 计算高斯核值
    kernel_value = np.exp(-distance_squared / (2 * sigma ** 2))
    return kernel_value

def func(arr):
    # 1d to 2d function
    length = len(arr)
    id = [i for i in range(length)]
    new = np.zeros((length, 2))
    new[:, 0] = np.array(id)
    new[:, 1] = np.array(arr)
    return new

def laplacian_kernel(x, y, sigma):
    # 计算x和y之间的曼哈顿距离
    manhattan_distance = np.sum(np.abs(x - y))
    # 计算拉普拉斯核
    return np.exp(-manhattan_distance / sigma)

def gaussian_kernel(x, y, sigma):
    # 计算x和y之间的欧氏距离
    euclidean_distance = np.linalg.norm(x - y)
    # 计算高斯核
    return np.exp(-(euclidean_distance ** 2) / (2 * (sigma ** 2)))

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
    
    # 初始化一个数组来保存每个点到各个中心的距离
    distances = np.zeros((len(points), len(centroids)))
    
    # 计算每个点与每个中心的距离
    for i, point in enumerate(points):
        for j, centroid in enumerate(centroids):
            if C == choose.DTW:
                distances[i, j] = similaritymeasures.dtw(func(point), func(centroid))[0]
            elif C == choose.L1:
                distances[i, j] = similaritymeasures.mae(point, centroid)
            elif C == choose.L2:
                distances[i, j] = similaritymeasures.mse(point, centroid)
            elif C == choose.Cos:
                distances[i, j] = 1 - spatial.distance.cosine(point, centroid)
            elif C == choose.GK:
                distances[i, j] = gaussian_kernel(point, centroid, sigma=1)
            elif C == choose.LK:
                distances[i, j] = laplacian_kernel(point, centroid, sigma=1)
            elif C == choose.IDK:
                distances[i, j] = np.dot(point, centroid)
            elif C == choose.GDK:
                distances[i, j] = gdk(point, centroid)
            elif C == choose.IK:
                distances[i, j] = np.dot(point, centroid)
            elif C == choose.FD:
                distances[i, j] = similaritymeasures.frechet_dist(func(point), func(centroid))
            elif C == choose.HD:
                 distances[i, j] = hausdorff_distance(func(point), func(centroid))
            elif C == choose.WD:
                distances[i, j] = wasserstein_distance(point, centroid)
                
    # 对于每个点，找到距离最小（相似度最大）的中心索引
    if C == choose.Cos or C == choose.GK or C == choose.LK or C == choose.IDK or C == choose.DTW or C == choose.GDK or C == choose.IK:    
        return np.argmax(distances, axis=1)
    
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

