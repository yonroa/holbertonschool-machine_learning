#!/usr/bin/env python3
"""Contains 'PCA' procedure"""
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.scatter(pca_data.T[0], pca_data.T[1], pca_data.T[2],
           c=labels, s=40, cmap=plt.cm.plasma)
ax.set_title("PCA of Iris Dataset")
ax.set_xlabel("U1")
ax.set_ylabel("U2")
ax.set_zlabel("U3")

plt.savefig("graph.png")
