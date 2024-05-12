import similaritymeasures
import numpy as np
import matplotlib.pyplot as plt
from hausdorff import hausdorff_distance
from scipy.stats import wasserstein_distance
from scipy import spatial


# Generate random experimental data
x = np.random.random(100)
y = np.random.random(100)
exp_data = np.zeros((100, 2))
exp_data[:, 0] = x
exp_data[:, 1] = y

# Generate random numerical data
x = np.random.random(100)
y = np.random.random(100)
num_data = np.zeros((100, 2))
num_data[:, 0] = x
num_data[:, 1] = y

# print("Experimental data:", exp_data)
# print("Numerical data:", num_data)


# quantify the difference between the two curves using PCM
pcm = similaritymeasures.pcm(exp_data, num_data) # 相似度

# quantify the difference between the two curves using
# Discrete Frechet distance
df = similaritymeasures.frechet_dist(exp_data, num_data) # 距离

# quantify the difference between the two curves using
# area between two curves
area = similaritymeasures.area_between_two_curves(exp_data, num_data) # 距离

# quantify the difference between the two curves using
# Curve Length based similarity measure
cl = similaritymeasures.curve_length_measure(exp_data, num_data) # 相似度

# quantify the difference between the two curves using
# Dynamic Time Warping distance
dtw, d = similaritymeasures.dtw(exp_data, num_data) # 相似度

# mean absolute error
mae = similaritymeasures.mae(exp_data, num_data) # 距离

# mean squared error
mse = similaritymeasures.mse(exp_data, num_data) # 距离

# hausdorff distance
hd = hausdorff_distance(exp_data, num_data) # 距离

# wasserstein distance distributional
wd = wasserstein_distance(exp_data[1], num_data[1]) # 距离

# cosine similarity
cos_sim = 1 - spatial.distance.cosine(exp_data[1], num_data[1]) # 相似度


# print the results
# print(pcm, df, area, cl, dtw, mae, mse, hd, wd, cos_sim)

# plot the data
plt.figure()
plt.plot(exp_data[:, 0], exp_data[:, 1])
plt.plot(num_data[:, 0], num_data[:, 1])
plt.show()