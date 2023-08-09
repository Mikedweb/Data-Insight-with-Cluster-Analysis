import numpy as np
from sklearn import datasets

N_SAMPLES = 1500

np.random.seed(0)

def circles():
    """Concentric circles
    
    Returns:
        ndarray (1500x2): data
    """

    data, _ = datasets.make_circles(n_samples=N_SAMPLES, factor=.5, noise=.05)
    return data

def moons():
    """Half-moons
    
    Returns:
        ndarray (1500x2): data
    """
    data, _ = datasets.make_moons(n_samples=N_SAMPLES, noise=.05)
    return data

def blobs():
    """Blobs, all with the same variance
    
    Returns:
        ndarray (1500x2): data
    """
    data, _ = datasets.make_blobs(n_samples=N_SAMPLES, random_state=17)
    return data

def random():
    """Randomly-generated data
    
    Returns:
        ndarray (1500x2): data
    """
    data = np.random.rand(N_SAMPLES, 2)
    return data

def anisotropic():
    """Skewed blobs
    
    Returns:
        ndarray (1500x2): data
    """
    data, _ = datasets.make_blobs(n_samples=N_SAMPLES, random_state=17)
    transform = [[0.6, -0.3], [-0.4, 0.8]]
    data = np.dot(data, transform)
    return data

def varied_variances():
    """Blobs with different variances
    
    Returns:
        ndarray (1500x2): data
    """
    data, _ = datasets.make_blobs(n_samples=N_SAMPLES, cluster_std=[1.0, 2.5, 0.5], random_state=17)
    return data

def iris():
    """The Iris Dataset
    
    Returns:
        dict: dataset and feature names
    """
    return datasets.load_iris()
