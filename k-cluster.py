import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


from datasets import(
    
    circles,
    moons,
    blobs,
    anisotropic,
    random,
    varied_variances
    )

X = blobs()


plt.figure()
plt.scatter(X[:,0], X[:,1])
plt.show()