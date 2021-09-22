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


def plot_MSE_vs_complexity(max_degree=19, n=1000, test_size=0.2, boot_resampling=False):
    """
    # TODO: docstrings
    """

    x_values, y_values, z_values = exercise1.generate_data(n, 0)

    # Scale data before further use
    x_values, y_values, z_values = exercise1.scaling_the_data(
        x_values, y_values, z_values)

    # Spliting in training and testing data
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x_values, y_values, z_values, test_size=test_size)

    list_of_MSE_testing = []
    list_of_BIAS_testing = []
    list_of_variance_testing = []
    list_of_MSE_training = []
    # TODO: comments
    for degree in range(1, max_degree + 1):
        betas, X_train = exercise1.get_betas_and_design_matrix(
            x_train, y_train, z_train, degree)

        # Find out how good the model is on our test data
        X_test = helper.create_design_matrix(x_test, y_test, degree)
        if boot_resampling:
            n_bootstrap = len(z_test)
            z_pred_test_matrix = np.empty((z_test.shape[0], n_bootstrap))

            # Running bootstrap-method on training data -> collect the different betas
            for i in range(n_bootstrap):
                _x, _y, _z = resample(x_train, y_train, z_train)

                # Evaluate the new model on the same test data each time.
                betas, _ = exercise1.get_betas_and_design_matrix(
                    _x, _y, _z, degree)
                z_pred_test_matrix[:, i] = exercise1.z_predicted(X_test, betas)

            # Take the mean of the betas from the bootstrap
            z_pred_test = np.mean(z_pred_test_matrix, axis=1, keepdims=True)
        else:
            z_pred_test = exercise1.z_predicted(X_test, betas)

        # Finding the mean of predicted z-values of the test
        # z_pred_test_mean = np.mean(z_pred_test)

        # Calculating the MSE, Bias, Variance
        # MSE_test = mean_squared_error(z_test, z_pred_test)
        # BIAS_test = np.mean((z_test - z_pred_test_mean) ** 2, keepdims=True)
        # # BIAS_test = np.mean((z_test - z_pred_test_matrix)**2)
        # variance_test = np.mean(

        MSE_test = np.mean(
            np.mean((z_test - z_pred_test_matrix)**2, axis=1, keepdims=True))

        BIAS_test = np.mean(
            (z_test - np.mean(z_pred_test_matrix, axis=1, keepdims=True))**2)
        variance_test = np.mean(
            np.var(z_pred_test_matrix, axis=1, keepdims=True))

        list_of_MSE_testing.append(MSE_test)
        list_of_BIAS_testing.append(BIAS_test)
        list_of_variance_testing.append(variance_test)

        # Find out how good the model is on our training data
        z_pred_train = exercise1.z_predicted(X_train, betas)
        MSE_train = mean_squared_error(z_train, z_pred_train)
        list_of_MSE_training.append(MSE_train)

    # TODO: nicer plots
    plt.plot(range(1, max_degree + 1),
             list_of_MSE_testing, label="Test sample - error")
    plt.plot(range(1, max_degree + 1),
             list_of_BIAS_testing, label="Test sample - bias")
    plt.plot(range(1, max_degree + 1),
             list_of_variance_testing, label="Test sample - variance")
    # plt.plot(range(1, max_degree + 1),
    #          list_of_MSE_training, label="Training sample")
    plt.xlabel("Model Complexity")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.title(
        f"MSE vs. complexity of the model\n\
        Bootstrap resampling: {boot_resampling}; Data points: {n}"
    )
    # plt.savefig(f"MSE_vs_Complexity_DP_{n}_BR_{boot_resampling}")
    plt.show()

    plt.close()


def main():
    plot_MSE_vs_complexity(max_degree=14, n=500, boot_resampling=True)
    # plot_MSE_vs_complexity(max_degree=10, n=100, boot_resampling=True)
    # plot_MSE_vs_complexity(max_degree=14, n=1000, boot_resampling=False)
    # plot_MSE_vs_complexity(max_degree=14, n=1000, boot_resampling=True)


if __name__ == '__main__':
    main()
