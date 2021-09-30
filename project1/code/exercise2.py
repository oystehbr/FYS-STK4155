from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import helper
import exercise1
import exercise4
import exercise5
import matplotlib.pyplot as plt
import numpy as np
# TODO scalar every column in design matrix ?? week 38


def bias_variance_boots(x_values, y_values, z_values, method, min_degree=1, max_degree=10, test_size=0.2, show_plot=False, lmbda=0.1):
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
    for degree in range(min_degree, max_degree + 1):
        # TODO: why len(z_test)
        n_bootstrap = len(z_test)
        z_pred_test_matrix = np.empty((z_test.shape[0], n_bootstrap))

        # TODO: comments
        X_test = helper.create_design_matrix(x_test, y_test, degree)
        # Running bootstrap-method on training data -> collect the different betas
        for i in range(n_bootstrap):
            # TODO: here loop over the different lambdas

            _x, _y, _z = resample(x_train, y_train, z_train)

            # Predicting z with the training set, _x, _y, _z
            z_pred_test, _, _ = helper.predict_output(
                x_train=_x, y_train=_y, z_train=_z,
                x_test=x_test, y_test=y_test,
                degree=degree, regression_method=method,
                lmbda=lmbda
            )

            z_pred_test_matrix[:, i] = z_pred_test

        # TODO: explain this, maybe vectorize
        MSE_test = 0
        n = len(z_test)
        for j in range(n):
            for i in range(n):

                MSE_test += (z_pred_test_matrix[j][i] - z_test[j])**2

        MSE_test *= 1/(n**2)

        # TODO: check bias
        BIAS_test = np.mean(
            (z_test - np.mean(z_pred_test_matrix, axis=1, keepdims=False))**2)

        variance_test = np.mean(
            np.var(z_pred_test_matrix, axis=1, keepdims=False))

        list_of_MSE_testing.append(MSE_test)
        list_of_BIAS_testing.append(BIAS_test)
        list_of_variance_testing.append(variance_test)

    if not show_plot:
        return list_of_MSE_testing, list_of_BIAS_testing, list_of_variance_testing

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
        Data points: {len(x_values)}; Method: {method}"
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

        z_pred_test, z_pred_train, _ = helper.predict_output(
            x_train=x_train, y_train=y_train, z_train=z_train,
            x_test=x_test, y_test=y_test,
            degree=degree, regression_method='OLS',
        )

        MSE_test = mean_squared_error(z_test, z_pred_test)
        list_of_MSE_testing.append(MSE_test)

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


def main(x_values, y_values, z_values, max_degree=8):

    # n = 200, noise = 0.0 - 0.1, max_degree = 8 -> great bias_variance
    bias_variance_boots(x_values, y_values, z_values, method="OLS",
                        max_degree=max_degree, test_size=0.2, show_plot=True)

    plot_MSE_vs_complexity(x_values, y_values, z_values,
                           max_degree=max_degree, test_size=0.2)


if __name__ == '__main__':
    n = 200
    noise = 0.05
    max_degree = 8
    x_values, y_values, z_values = helper.generate_data(n, noise)
    main(x_values, y_values, z_values, max_degree)
