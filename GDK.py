import numpy as np
import scipy.io as io
def gdk(S, T):
    length = len(S)
    score = 0.0
    for x in S:
        for y in T:
            score += gaussian_kernel(x, y)
    return score / (length * length)

def gaussian_kernel(x, y, sigma=1.0):
    # 计算样本之间的欧氏距离的平方
    distance_squared = np.sum((x - y) ** 2)
    # 计算高斯核值
    kernel_value = np.exp(-distance_squared / (2 * sigma ** 2))
    return kernel_value

if __name__=='__main__':
    raw_mat = io.loadmat("./datasets/TRAFFIC.mat")
    # data_ = raw_mat["tracks_traffic"]
    # label_ = raw_mat["truth"]
    print(raw_mat)
    # 示例
    # x = np.array([1, 2, 3])
    # y = np.array([4, 5, 6])
    # print("Gaussian distribution Kernel", gdk(x, y))