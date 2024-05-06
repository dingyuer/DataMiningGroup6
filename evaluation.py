# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:28:13 2019

@author: liuxin
"""

import numpy as np
import matplotlib.pyplot as plt
from regression import linear_scores, ridge_scores, lasso_scores, elasticNet_scores

"""
    构造降序排序矩阵
"""
def rank_matrix(matrix):
    cnum = matrix.shape[1]
    rnum = matrix.shape[0]
    ## 升序排序索引
    sorts = np.argsort(matrix)
    for i in range(rnum):
        k = 1
        n = 0
        flag = False
        nsum = 0
        for j in range(cnum):
            n = n+1
            ## 相同排名评分序值
            if j < 3 and matrix[i, sorts[i,j]] == matrix[i, sorts[i,j + 1]]:
                flag = True;
                k = k + 1;
                nsum += j + 1;
            elif (j == 3 or (j < 3 and matrix[i, sorts[i,j]] != matrix[i, sorts[i,j + 1]])) and flag:
                nsum += j + 1
                flag = False;
                for q in range(k):
                    matrix[i,sorts[i,j - k + q + 1]] = nsum / k
                k = 1
                flag = False
                nsum = 0
            else:
                matrix[i, sorts[i,j]] = j + 1
                continue
    return matrix

"""
    Friedman检验
    参数：数据集个数n, 算法种数k, 排序矩阵rank_matrix(k x n)
    函数返回检验结果（对应于排序矩阵列顺序的一维数组）
"""
def friedman(n, k, rank_matrix):
    # 计算每一列的排序和
    sumr = sum(list(map(lambda x: np.mean(x) ** 2, rank_matrix.T)))
    result = 12 * n / (k * ( k + 1)) * (sumr - k * (k + 1) ** 2 / 4)
    result = (n - 1) * result /(n * (k - 1) - result)
    return result

"""
    Nemenyi检验
    参数：数据集个数n, 算法种数k, 排序矩阵rank_matrix(k x n)
    函数返回CD值
"""

def nemenyi(n, k, q):
    return q * (np.sqrt(k * (k + 1) / (6 * n)))

qa = [0, 0, 1.96, 2.344, 2.569, 2.728, 2.85, 2.949, 3.031, 3.102, 3.164]
"""
输入数据: 
    -matrix 测试结果矩阵（行为各数据集，列为各算法）
    -n 数据集个数
    -k 算法个数
    -name 算法名列表
"""
def Friedman_test(matrix, n, k, name = ["linear_scores", "ridge_scores", "lasso_scores", "elasticNet_scores"]):
    matrix_r = rank_matrix(matrix)
    Friedman = friedman(n, k, matrix_r)
    CD = nemenyi(n, k, qa[k])
    rank_x = list(map(lambda x: np.mean(x), matrix.T))
    name_y = name
    min_ = [x for x in rank_x - CD/2]
    max_ = [x for x in rank_x + CD/2]

    plt.figure(figsize=(12, 5)) #700*700像素
    plt.title("Friedman")
    plt.scatter(rank_x, name_y)
    plt.hlines(name_y, min_, max_)
    plt.show()

if __name__=="__main__":
    matrix = np.array([linear_scores, ridge_scores, lasso_scores, elasticNet_scores])
    Friedman_test(matrix.T, 10, 4)