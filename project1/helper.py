"""
All valuable functions will be added here

"""
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: comments, dogstring, and  make code to our own


def create_design_matrix(x, y, n):
    """
    # TODO: create docstring
    """

    if len(x.shape) > 1:
        x = np.ravel(x)  # TODO: hva gjør den
        y = np.ravel(y)

    col_no = len(x)
    row_no = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((col_no, row_no))

    for i in range(1, n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)

    return X

# TODO: train_test_split in SciKit-Learn
