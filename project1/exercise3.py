import exercise2
import exercise4
import exercise5
import numpy as np
import exercise1
import helper
import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt
from sklearn.utils import resample
import exercise1
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def _get_test_and_train_block(values: np.ndarray, i: int, test_no: int, last: bool = False):
    """
    Function that gives out the i-th block of test-data, and gives
    the remaining as training-data

    Usage: for dividing the data in cross-validation

    :param values (np.ndarray):
        the values you wanna get blocks of data of
    :param i (int):
        indicate the block_no, we want to get
    :param test_no (int):
        amount of test data

    :return (tuple):
        train and test data
    """

    if not last:
        train = np.concatenate((
            values[0:i*test_no],
            values[(i+1)*test_no:]
        ))
        test = values[i*test_no:(i+1)*test_no]
    else:
        train = values[0:i*test_no]
        test = values[i*test_no:]

    return train, test


def cross_validation(x_values, y_values, z_values, method, k_folds=5, degree=5, lmbda=0.1):
    """
    # TODO: docs

    """
    # TODO: x_values and y_values together? INTO X, and better normalizing data
    if len(x_values) % k_folds > 2:
        test_no = len(x_values) // k_folds + 1
    else:
        test_no = len(x_values) // k_folds

    last = False
    MSE_list = np.zeros(k_folds)
    for i in range(k_folds):
        if i == (k_folds - 1):
            last = True

        x_train, x_test = _get_test_and_train_block(x_values, i, test_no, last)
        y_train, y_test = _get_test_and_train_block(y_values, i, test_no, last)
        z_train, z_test = _get_test_and_train_block(z_values, i, test_no, last)

        if False:
            # Test for checking that it gives the correct output
            print(f'ROUND {i}')
            print(x_values)
            print(f'x_train: {x_train}, len: {len(x_train)} ')
            print(f'x_test: {x_test}, len: {len(x_test)} ')
            print('------------------------')

        # TODO: what do we want with this
        # Train the model with the training data
        X_train = helper.create_design_matrix(x_train, y_train, degree)
        X_train_scale = np.mean(X_train, axis=0)
        X_train_scaled = X_train - X_train_scale

        z_train_scale = np.mean(z_train, axis=0)
        z_train_scaled = z_train - z_train_scale

        X_test = helper.create_design_matrix(x_test, y_test, degree)
        X_test_scaled = X_test - X_train_scale

        # Evaluate the new model on the same test data each time.
        if method == 'OLS':
            betas = exercise1.get_betas_OLS(
                X_train_scaled, z_train_scaled)
        elif method == 'RIDGE':
            betas = exercise4.get_betas_RIDGE(
                X_train_scaled, lmbda, z_train_scaled)
        elif method == 'LASSO':
            betas = exercise5.get_betas_LASSO(
                X_train_scaled, lmbda, z_train_scaled)
        else:
            print("incorrenct regression model in cross_validation")

        # Finding the predicted z values with the current model
        z_pred = exercise1.z_predicted(
            X_test_scaled, betas) + z_train_scale

        MSE_list[i] = exercise1.mean_squared_error(z_test, z_pred)

    estimated_MSE_cross_validation = np.mean(MSE_list)
    estimated_MSE_bootstrap = exercise2.bias_variance_boots(
        x_values, y_values, z_values, method, min_degree=degree, max_degree=degree, lmbda=lmbda)[0][-1]

    X = helper.create_design_matrix(x_values, y_values, degree)
    # estimated_mse_sckit_list_neg = cross_val_score(
    #     linear_model.LinearRegression(), X[:, np.newaxis], z_values[:, np.newaxis], scoring='neg_mean_squared_error', cv=5)
    # estimated_mse_sckit = np.mean(-estimated_mse_sckit_list_neg)
    return estimated_MSE_cross_validation, estimated_MSE_bootstrap


def main():
    n = 100
    x_values, y_values, z_values = exercise1.generate_data(n, 0)
    MSE_cross_validation, MSE_bootstrap = cross_validation(x_values, y_values, z_values,
                                                           method='OLS',  k_folds=5, degree=10)
    print(f'MSE_cross: {MSE_cross_validation}')
    print(f'MSE_boot: {MSE_bootstrap}')


if __name__ == "__main__":
    main()
