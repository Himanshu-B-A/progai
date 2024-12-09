from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv(r"C:\Users\himub\OneDrive\Desktop\PROG\Labprog_AI AND ML\iris.csv", header=0)

# Map class labels to numeric values
label_map = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
y = [label_map[c] for c in dataset['variety']]  # Update 'variety' to actual column name if different

# Extract features
X = dataset[['petal.length', 'petal.width']]  # Using only Petal Length and Petal Width for simplicity

# Create colormap
colormap = np.array(['red', 'lime', 'black'])

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=3425).fit(X)
kmeans_labels = kmeans.labels_

# Gaussian Mixture Model Clustering
gmm = GaussianMixture(n_components=3, random_state=3425).fit(X)
gmm_labels = gmm.predict(X)

# Plot results
plt.figure(figsize=(14, 7))

# Real Data Plot
plt.subplot(1, 3, 1)
plt.title('Real Data')
plt.scatter(X['petal.length'], X['petal.width'], c=colormap[y])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# KMeans Clustering
plt.subplot(1, 3, 2)
plt.title('KMeans Clustering')
plt.scatter(X['petal.length'], X['petal.width'], c=colormap[kmeans_labels])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# GMM Clustering
plt.subplot(1, 3, 3)
plt.title('GMM Clustering')
plt.scatter(X['petal.length'], X['petal.width'], c=colormap[gmm_labels])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# Show plots
plt.tight_layout()
plt.show()

# Evaluate KMeans and GMM
print('KMeans Accuracy: ', metrics.accuracy_score(y, kmeans_labels))
print('KMeans Confusion Matrix:\n', metrics.confusion_matrix(y, kmeans_labels))

print('GMM Accuracy: ', metrics.accuracy_score(y, gmm_labels))
print('GMM Confusion Matrix:\n', metrics.confusion_matrix(y, gmm_labels))
