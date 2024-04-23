import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

tsne = TSNE(perplexity=2, n_components=2)
x = [[1, 2, 2, 5, 6, 2, 4], [2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3, 3]]
y = [1, 0, 2]  # y是x对应的标签
x = np.array(x)
y = np.array(y)
x_tsne = tsne.fit_transform(x)
plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y)
plt.show()


'''
x = np.array(x)
y = np.array(y)
'''