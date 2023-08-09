#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# read in our data
cc_data = pd.read_csv('CC_GENERAL.csv', index_col=False)
#print(cc_data.isnull().sum())

# plot CREDIT_LIMIT v. BALANCE
#cc_data.plot.scatter(x='CREDIT_LIMIT', y='BALANCE')
#cc_data.sample(1000).plot.scatter(x='CREDIT_LIMIT', y='BALANCE')
#plt.show()

subsample = cc_data.sample(1000, random_state=17)
X = pd.concat([subsample['CREDIT_LIMIT'], subsample['BALANCE']], axis=1)

# draw a graph for the elbow method
"""
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.scatter(range(1, 11), wcss)
plt.show()
"""

# train KMeans on X and plot with colors denoting clusters
"""
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
subsample.plot.scatter(x='CREDIT_LIMIT', y='BALANCE', c=kmeans.labels_, colormap='viridis')
"""

hac = AgglomerativeClustering(n_clusters=4, linkage='ward')
hac.fit(X)
subsample.plot.scatter(x='CREDIT_LIMIT', y='BALANCE', c=hac.labels_, colormap='viridis')

plt.axis('equal')
plt.show()
