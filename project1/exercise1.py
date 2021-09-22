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
from sklearn.model_selection import train_test_split
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def generate_data(n, noise_multiplier=0.1):
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


def get_betas_and_design_matrix(x_values, y_values, z_values, degree=2):
    """
    TODO: docstrings


    """

    # Creating the design matrix for our x- and y-values
    X = helper.create_design_matrix(x_values, y_values, degree)

    X_T = np.matrix.transpose(X)
    betas = np.linalg.inv(X_T @ X) @ X_T @ z_values
    # LÆRER SA DENNE VAR BEST:
    # beta = np.linalg.pinv(X_T @ X) @ X_T @ z_values -> SVD
    return betas, X


def find_variance(z_true, z_pred):
    """
    # TODO: docs
    """

    n = len(z_true)
    sum = 0
    for t, p in zip(z_true, z_pred):
        sum += (t - p)**2
    return sum/n


def get_confidence_interval_ND(betas, X, z_true, CI_num=0.95):
    """
    Function that calculate the confidence interval for the given bates
    returns the interval as a list

    # TODO: docstrings

    """

    # TODO: sigma_squared found right
    z_pred = z_predicted(X, betas)
    sigma_squared = find_variance(z_true, z_pred)

    X_T = np.matrix.transpose(X)
    cov_matrix = sigma_squared * np.linalg.inv(X_T @ X)

    # TODO: calculate z, w.r.t CI_num
    z = 1.96

    n = len(z_true)
    list_of_confidence_intervals = []
    for i, mean_beta in enumerate(betas):
        sigma_beta = np.sqrt(sigma_squared * cov_matrix[i][i])
        CI = [mean_beta - z*sigma_beta,
              mean_beta + z*sigma_beta]
        list_of_confidence_intervals.append(CI)

    return list_of_confidence_intervals


def print_betas_CI(betas, CI_list):

    print(f'Beta (No)  Beta-value  CI')
    for i, (beta, CI) in enumerate(zip(betas, CI_list)):
        print(f'{i} | {beta} | {CI}')
        return


def z_predicted(X, betas):
    """
    Returns a  list of z-values predicted by the regression model

    # TODO: docstrings
    """

    return X @ betas


def scaling_the_data(x, y, z):
    """
    # TODO: docs, better scaling, option

    """

    return x - np.mean(x), y - np.mean(y), z - np.mean(z)


def main():
    # TODO: variables in main() -> constant over main()
    n = 1000
    degree = 10
    test_size = 0.2

    x_values, y_values, z_values = generate_data(n, 0.1)
    # Scale data before further use
    x_values, y_values, z_values = scaling_the_data(
        x_values, y_values, z_values)

    # We split the data in test and training data
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x_values, y_values, z_values, test_size=test_size)

    # Train the model with the training data
    betas, X_train = get_betas_and_design_matrix(
        x_train, y_train, z_train, degree)

    # Find the confidence intervals of the betas
    CI_list = get_confidence_interval_ND(
        betas, X_train, z_train)

    # print_betas_CI(betas, CI_list)

    # TODO: create function for getting MSE and R2_score at same time
    # Evaluating the Mean Squared error (MSE)
    X_test = helper.create_design_matrix(x_test, y_test, degree)
    z_pred = z_predicted(X_test, betas)

    MSE = mean_squared_error(z_test, z_pred)
    R2_score = r2_score(z_test, z_pred)

    print(f'MSE: {MSE}')
    print(f'R2_score: {R2_score}')

    # Evaluating the R^2 score function


if __name__ == "__main__":
    main()
