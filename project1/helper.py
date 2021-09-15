"""
All valuable functions will be added here

"""
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: comments, dogstring, and  make code to our own


def create_design_matrix(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)

    return X

# TODO: train_test_split in SciKit-Learn
