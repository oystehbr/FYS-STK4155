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
from sklearn.utils import resample
import exercise1
# TODO scalar every column in design matrix ?? week 38


def bias_variance_boots(x_values, y_values, z_values, max_degree=10, test_size=0.2, show_plot=False):
    """
    # TODO: better name

    """

    # Spliting in training and testing data
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x_values, y_values, z_values, test_size=test_size)

    list_of_MSE_testing = []
    list_of_BIAS_testing = []
    list_of_variance_testing = []
    # TODO: comments
    for degree in range(1, max_degree + 1):
        # TODO: why len(z_test)
        n_bootstrap = len(z_test)
        z_pred_test_matrix = np.empty((z_test.shape[0], n_bootstrap))

        # TODO: comments
        X_test = helper.create_design_matrix(x_test, y_test, degree)
        X_test_scaled = exercise1.scale_design_matrix(X_test)
        # Running bootstrap-method on training data -> collect the different betas
        for i in range(n_bootstrap):

            _x, _y, _z = resample(x_train, y_train, z_train)

            # Train the model with the training data
            X_train = helper.create_design_matrix(_x, _y, degree)
            X_train_scaled = exercise1.scale_design_matrix(X_train)

            z_train_scaled = exercise1.scale_design_matrix(_z)

            # Evaluate the new model on the same test data each time.
            betas_OLS = exercise1.get_betas_OLS(
                X_train_scaled, z_train_scaled)

            # Finding the predicted z values with the current model
            z_pred = exercise1.z_predicted(
                X_test_scaled, betas_OLS) + np.mean(_z)

            z_pred_test_matrix[:, i] = z_pred.ravel()

        MSE_test = np.mean(
            np.mean((z_test - z_pred_test_matrix)**2, axis=1, keepdims=True))

        # TODO: check bias
        BIAS_test = np.mean(
            (z_test - np.mean(z_pred_test_matrix, axis=1, keepdims=True))**2)

        variance_test = np.mean(
            np.var(z_pred_test_matrix, axis=1, keepdims=True))

        list_of_MSE_testing.append(MSE_test)
        list_of_BIAS_testing.append(BIAS_test)
        list_of_variance_testing.append(variance_test)

    if not show_plot:
        return list_of_MSE_testing

    # TODO: nicer plots
    plt.plot(range(1, max_degree + 1),
             list_of_MSE_testing, label="Test sample - error")
    plt.plot(range(1, max_degree + 1),
             list_of_BIAS_testing, label="Test sample - bias")
    plt.plot(range(1, max_degree + 1),
             list_of_variance_testing, label="Test sample - variance")
    plt.xlabel("Model Complexity")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.title(
        f"MSE vs. complexity of the model\n\
        Bootstrap resampling: True; Data points: {len(x_values)}"
    )
    # plt.savefig(f"MSE_vs_Complexity_DP_{n}_BR_{boot_resampling}")
    plt.show()
    plt.close()


def plot_MSE_vs_complexity(x_values, y_values, z_values, max_degree=10, test_size=0.2):
    """
    # TODO: docstrings
    # TODO: better function_name
    """

    # Spliting in training and testing data
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x_values, y_values, z_values, test_size=test_size)

    list_of_MSE_testing = []
    list_of_MSE_training = []
    # Finding MSE for training and test data for different degrees
    for degree in range(1, max_degree + 1):
        # Get designmatrix from the training data and scale it
        X_train = helper.create_design_matrix(x_train, y_train, degree)
        X_train_scaled = exercise1.scale_design_matrix(X_train)

        # Get designmatrix from the test data and scale it
        X_test = helper.create_design_matrix(x_test, y_test, degree)
        X_test_scaled = exercise1.scale_design_matrix(X_test)

        # Scale the output_values
        z_train_scaled = exercise1.scale_design_matrix(z_train)

        # Get the betas from OLS - method
        betas_OLS = exercise1.get_betas_OLS(X_train_scaled, z_train_scaled)

        # Find out how good the model is on our test data
        X_test = helper.create_design_matrix(x_test, y_test, degree)

        # Find out how good the model is on our test data
        z_pred_test = exercise1.z_predicted(
            X_test_scaled, betas_OLS) + np.mean(z_train)
        MSE_test = mean_squared_error(z_test, z_pred_test)
        list_of_MSE_testing.append(MSE_test)

        # Find out how good the model is on our training data
        z_pred_train = exercise1.z_predicted(
            X_train_scaled, betas_OLS) + np.mean(z_train)
        MSE_train = mean_squared_error(z_train, z_pred_train)
        list_of_MSE_training.append(MSE_train)

    # TODO: nicer plots
    plt.plot(range(1, max_degree + 1),
             list_of_MSE_testing, label="Test sample - error")
    plt.plot(range(1, max_degree + 1),
             list_of_MSE_training, label="Training sample")
    plt.xlabel("Model Complexity")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.title(
        f"MSE vs. complexity of the model\nData points: {len(x_values)}"
    )
    # plt.savefig(f"MSE_vs_Complexity_DP_{n}")
    plt.show()
    plt.close()


def main(n=20, noise=0.1):

    x_values, y_values, z_values = exercise1.generate_data(n, noise)

    bias_variance_boots(x_values, y_values, z_values,
                        max_degree=5, test_size=0.2)

    # plot_MSE_vs_complexity(x_values, y_values, z_values,
    #                        max_degree=20, test_size=0.2)


if __name__ == '__main__':
    main()
