from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Data
x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 6, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
# Plot the original data
plt.scatter(x1, x2)
plt.xlim()
plt.ylim()
plt.title('Dataset')
plt.show()
# KMeans clustering
X = np.array([x1, x2]).T
kmeans = KMeans(n_clusters=3).fit(X)
# Plot clustered data
plt.scatter(x1, x2, c=kmeans.labels_)
plt.xlim()
plt.ylim()
plt.title('KMeans Clustering')
plt.show()
