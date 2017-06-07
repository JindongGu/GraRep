
'''
This is an implementation of K Means Clustering Algorithm.

The function used to mearsure distance between feature vectors is cosine similarity function.

Requires : python 2.7.x, Numpy 1.7.1+ 
'''

import numpy as np
from scipy.spatial.distance import cosine
from numpy.linalg import norm

def dist(x_diff, y_diff):
    if norm(x_diff) == 0 and norm(y_diff) == 0:
        return 0
    elif norm(x_diff) == 0 or norm(y_diff) == 0:
        return 1        
    else:
        return cosine(x_diff, y_diff)


def kMeans(X, K, maxIters = 200):

    if K > len(X):
        raise Exception('specify too many Clusters!!!')

    #initialise the cluster centers
    centroids = X[np.random.choice(np.arange(len(X)), K), :]

    for i in range(maxIters):
        
        # Cluster Assignment step
        C = np.array([np.argmin([dist(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        
        # Move centroids step
        centroids = np.asarray([centroids[k] if (np.any(X[C == k]) == False) else X[C == k].mean(axis = 0) for k in range(K)])
         
    return np.array(centroids) , C
