#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from datasets import (
    circles,
    moons,
    blobs,
    anisotropic,
    random,
    varied_variances
)

X = varied_variances()
dbscan = DBSCAN(eps=1, min_samples=5)

dbscan.fit(X)

# get inliers and their cluster
X_inlier = X[dbscan.labels_ != -1]
y_inlier = dbscan.labels_[dbscan.labels_ != -1]

# get outliers
X_outlier = X[dbscan.labels_ == -1]

plt.scatter(X_inlier[:,0], X_inlier[:,1], c=y_inlier, cmap='Dark2')
plt.scatter(X_outlier[:,0], X_outlier[:,1], c='k')
plt.show()
