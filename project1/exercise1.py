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
        x_array[i] = np.random.uniform(0, 1)
        y_array[i] = np.random.uniform(0, 1)
        eps = np.random.normal(0, 1)
        data_array[i] = franke_function(
            x_array[i], y_array[i]) + noise_multiplier * eps

    return x_array, y_array, data_array


def get_betas_OLS(X, z_values, degree=2):
    """
    TODO: docstrings


    """

    X_T = np.matrix.transpose(X)
    betas = np.linalg.pinv(X_T @ X) @ X_T @ z_values
    # LÆRER SA DENNE VAR BEST:
    # TODO: sjekk denne metoden for å regne betas
    # beta = np.linalg.pinv(X_T @ X) @ X_T @ z_values -> SVD
    return betas


def scale_design_matrix(X, mean_scale):
    """
    Scaling the desingmatrix by subtracting the mean of each column

    :param X (np.ndarray): # TODO: check type
        the matrix we wanna scale 

    :return:
        the scaled matrix
    """

    X_scaled = X - mean_scale
    return X_scaled


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
    # TODO: explain what happens here
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


def z_predicted(X, betas):
    """
    Returns a  list of z-values predicted by the regression model

    # TODO: docstrings
    """

    return X @ betas


def main(n=1000, degree=5, test_size=0.2, noise=0):

    # TODO: test with scikit learn -> week 38 lectures

    x_values, y_values, z_values = generate_data(n, noise)
    # Scale data before further use

    # We split the data in test and training data
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x_values, y_values, z_values, test_size=test_size)

    # Train the model with the training data
    X_train = helper.create_design_matrix(x_train, y_train, degree)
    X_test = helper.create_design_matrix(x_test, y_test, degree)

    # Scale data before further use
    X_train_scale = np.mean(X_train, axis=0)
    X_train_scaled = X_train - X_train_scale

    # TODO: shall we scale with the above?
    X_test_scaled = X_test - X_train_scale
    z_train_scale = np.mean(z_train, axis=0)
    z_train_scaled = z_train - z_train_scale

    # Get the betas from OLS.
    betas_OLS = get_betas_OLS(X_train_scaled, z_train_scaled)

    # Find the confidence intervals of the betas # TODO: confidence interval scaled
    CI_list = get_confidence_interval_ND(
        betas_OLS, X_train, z_train)

    # print_betas_CI(betas_OLS, CI_list)

    # Scale the data back to its original form
    z_pred_test = z_predicted(X_test_scaled, betas_OLS) + z_train_scale

    # Evaluating the Mean Squared error (MSE)
    # TODO: create function for getting MSE and R2_score at same time
    MSE = mean_squared_error(z_test, z_pred_test)
    R2_score = r2_score(z_test, z_pred_test)

    print(f'MSE: {MSE}')
    print(f'R2_score: {R2_score}')

    # Evaluating the R^2 score function


if __name__ == "__main__":
    main()
