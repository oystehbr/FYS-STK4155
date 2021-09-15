import helper
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import random
import numpy as np


def franke_function(x, y):
    """
    Compute and return function value for a Franke's function

    :param x (float):
        input value
    :param y (float):
        input value
    :return (float):
        function value 
    """

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def generate_data(n, noise_multiplier=1):
    """ 
    Generates data
    :param n (int):
        number of x and y values
    :param noise_multiplier (float, int):
        scale the noise
    :return (np.ndarray):
        array of generated funciton values with noise
    """

    # TODO: vectorize
    data_array = np.zeros(n)
    x_array = np.zeros(n)
    y_array = np.zeros(n)
    for i in range(n):
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        eps = np.random.normal(0, 1)
        z = franke_function(x, y) + noise_multiplier * eps
        x_array[i] = x
        y_array[i] = y
        data_array[i] = z
    return x_array, y_array, data_array


def get_betas(x_values, y_values, z_values):
    """
    TODO: docstrings
    X_transformed's form for 2-th order polynomial(MATRIX), x = > m*2-matrix:
    1 x y x**2 x*y y**2
    """
    # design matrix:

    m = len(x_values)
    X = np.zeros((m, 6))

    # Adding the columns of the design matrix
    # TODO: call helper function with design matrix
    X[:, 0] = 1
    X[:, 1] = x_values
    X[:, 2] = y_values
    X[:, 3] = x_values**2
    X[:, 4] = x_values * y_values
    X[:, 5] = y_values**2

    X_T = np.matrix.transpose(X)
    beta = np.linalg.inv(X_T @ X) @ X_T @ z_values
    # beta = np.linalg.pinv(X_T @ X) @ X_T @ z_values -> SVD
    return beta


def main():
    n = 10000
    x_values, y_values, z_values = generate_data(n, 0)

    betas = get_betas(x_values, y_values, z_values)
    print(betas)

#  f(x, y) = Dersom(0 < x < 1 ∧ 0 < y < 1, 1.19606753 - -1.02899952x + -0.75610173y + 0.06874209x² + 0.88365586x*y - -0.38809648y²)


if __name__ == "__main__":
    main()
