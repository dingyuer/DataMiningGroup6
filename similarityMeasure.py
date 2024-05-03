import numpy as np
"""
参数:
    - x, y: numpy数组，表示特征空间中的两个向量。
    - sigma: 核函数的尺度参数，控制衰减的速度。
"""

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

    