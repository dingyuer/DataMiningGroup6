from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn import metrics

def NMI_example():
    labels_true = [0, 0, 1, 1]
    labels_pred = [0, 0, 1, 1]

    score = normalized_mutual_info_score(labels_true, labels_pred)
    print(score) # 1.0

    labels_true = [0, 0, 0, 0]
    labels_pred = [0, 1, 2, 3]

    score = normalized_mutual_info_score(labels_true, labels_pred)
    print(score) # 0.0

def ARI_example():
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

if __name__== '__main__':
    NMI_example()
    ARI_example()