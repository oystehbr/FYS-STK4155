import helper
import exercise1
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
import matplotlib.pyplot as plt


def plot_MSE_vs_complexity():
    """
    # TODO: docstrings
    """

    max_degree = 19
    n = 1000
    test_size = 0.2

    x_values, y_values, z_values = exercise1.generate_data(n, 0.1)
    # Scale data before further use
    x_values, y_values, z_values = exercise1.scaling_the_data(
        x_values, y_values, z_values)

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x_values, y_values, z_values, test_size=test_size)

    list_of_MSE_testing = []
    list_of_MSE_training = []
    for degree in range(1, max_degree + 1):
        betas, X = exercise1.get_betas_and_design_matrix(
            x_train, y_train, z_train, degree)

        # Sample testing
        X_pred = helper.create_design_matrix(x_test, y_test, degree)
        z_pred_test = exercise1.z_predicted(X_pred, betas)
        MSE_test = mean_squared_error(z_test, z_pred_test)
        list_of_MSE_testing.append(MSE_test)

        # Sample training
        z_pred_train = exercise1.z_predicted(X, betas)
        MSE_train = mean_squared_error(z_train, z_pred_train)
        list_of_MSE_training.append(MSE_train)

    # TODO: nicer plots
    plt.plot(range(1, max_degree + 1),
             list_of_MSE_testing, label="Test sample")
    plt.plot(range(1, max_degree + 1),
             list_of_MSE_training, label="Training sample")
    plt.xlabel("Model Complexity")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.title("Mean squared error vs. complexity of the model")
    plt.show()


def main():
    plot_MSE_vs_complexity()


if __name__ == '__main__':
    main()
