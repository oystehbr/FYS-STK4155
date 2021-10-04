import helper
import matplotlib.pyplot as plt
import numpy as np


def bias_variance_boots_looping_lambda(x_values, y_values, z_values, method, degree, n_bootstrap=100, test_size=0.2, lmbda=0.1):
    """
    Function for calculating and return the MSE, bias and variance (testing data).
    Will be used for Ridge and Lasso regression. 

    :param x_values (np.ndarray):
        dependent variable
    :param y_values (np.ndarray):
        dependent variable
    :param z_values (np.ndarray):
        response variable
    :param method (str):
        the preffered regression method: OLS, RIDGE or LASSO
    :param degree (int):
        the order of the polynomial that defines the design matrix
    :param n_bootstrap (int):
        the number of bootstrap iterations
    :param test_size (float)
        the amount of data we will use in testing
    :param lmbda (float):
        parameter used by Ridge and Lasso regression (lambda)

    :return (None, tuple[list, list, list]):
        - MSE for the given model
        - bias for the given model
        - variance for the given model 
    """

    # Spliting in training and testing data
    x_train, x_test, y_train, y_test, z_train, z_test = helper.train_test_split(
        x_values, y_values, z_values, test_size=test_size)

    # Store the MSE, bias and variance values for test data
    list_of_MSE_testing = []
    list_of_bias_testing = []
    list_of_variance_testing = []

    z_pred_test_matrix = np.empty((z_test.shape[0], n_bootstrap))

    # Running bootstrap-method
    for i in range(n_bootstrap):

        _x, _y, _z = helper.resample(x_train, y_train, z_train)

        # Get the predicted values given our test- and training data
        z_pred_test, _, _ = helper.predict_output(
            x_train=_x, y_train=_y, z_train=_z,
            x_test=x_test, y_test=y_test,
            degree=degree, regression_method=method,
            lmbda=lmbda
        )

        z_pred_test_matrix[:, i] = z_pred_test

        # Finding MSE, bias and variance from the bootstrap
        MSE_test = np.mean(np.mean(
            (z_pred_test_matrix - np.transpose(np.array([z_test])))**2))

        bias_test = np.mean(
            (z_test - np.mean(z_pred_test_matrix, axis=1, keepdims=False))**2)

        variance_test = np.mean(
            np.var(z_pred_test_matrix, axis=1, keepdims=False))

        list_of_MSE_testing.append(MSE_test)
        list_of_bias_testing.append(bias_test)
        list_of_variance_testing.append(variance_test)

    return list_of_MSE_testing, list_of_bias_testing, list_of_variance_testing


def bias_variance_boots_looping_degree(x_values, y_values, z_values, method, max_degree=10, n_bootstrap=100, test_size=0.2, show_plot=False, lmbda=0.1):
    """
    Function for plotting/ or returning the MSE, bias and variance (testing data), 
    over complexity. Where complexity means the order of the polynomial 
    that defines the design matrix

    :param x_values (np.ndarray):
        dependent variable
    :param y_values (np.ndarray):
        dependent variable
    :param z_values (np.ndarray):
        response variable
    :param method (str):
        the preffered regression method: OLS, RIDGE or LASSO
    :param max_degree (int):
        the maximum order of the polynomial that defines the design matrix
    :param n_bootstrap (int):
        the number of bootstrap iterations
    :param test_size (float)
        the amount of data we will use in testing
    :param show_plot (bool):
        - True: plot will show
        - False: will return lists of MSE, bias and variance
    :param lmbda (float):
        parameter used by Ridge and Lasso regression (lambda)

    :return (None, tuple[list, list, list]):
        - MSE for the different degrees
        - bias for the different degrees
        - variance for the different degrees
    """

    # Spliting in training and testing data
    x_train, x_test, y_train, y_test, z_train, z_test = helper.train_test_split(
        x_values, y_values, z_values, test_size=test_size)

    # Store the MSE, bias and variance values for test data
    list_of_MSE_testing = []
    list_of_bias_testing = []
    list_of_variance_testing = []

    # Finding MSE, bias and variance of test data for different degrees
    for degree in range(1, max_degree + 1):

        # Create matrix for storing the bootstrap results
        z_pred_test_matrix = np.empty((z_test.shape[0], n_bootstrap))

        # Running bootstrap-method
        for i in range(n_bootstrap):

            _x, _y, _z = helper.resample(x_train, y_train, z_train)

            # Predicting z with the training set, _x, _y, _z
            z_pred_test, _, _ = helper.predict_output(
                x_train=_x, y_train=_y, z_train=_z,
                x_test=x_test, y_test=y_test,
                degree=degree, regression_method=method,
                lmbda=lmbda
            )

            z_pred_test_matrix[:, i] = z_pred_test

        # Finding MSE, bias and variance from the bootstrap
        MSE_test = np.mean(np.mean(
            (z_pred_test_matrix - np.transpose(np.array([z_test])))**2))

        bias_test = np.mean(
            (z_test - np.mean(z_pred_test_matrix, axis=1, keepdims=False))**2)

        variance_test = np.mean(
            np.var(z_pred_test_matrix, axis=1, keepdims=False))

        list_of_MSE_testing.append(MSE_test)
        list_of_bias_testing.append(bias_test)
        list_of_variance_testing.append(variance_test)

    if not show_plot:
        return list_of_MSE_testing, list_of_bias_testing, list_of_variance_testing

    plt.plot(range(1, max_degree + 1),
             list_of_MSE_testing, label="Test sample - error")
    plt.plot(range(1, max_degree + 1),
             list_of_bias_testing, label="Test sample - bias")
    plt.plot(range(1, max_degree + 1),
             list_of_variance_testing, label="Test sample - variance")
    plt.xlabel("Model Complexity")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.title(
        f"bias-variance trade-off vs model complexity\n\
        Data points: {len(x_values)}; Method: {method}"
    )
    plt.savefig(
        f"figures/bias_variance_boots_looping_degree_DP_{len(x_values)}_d_{degree}_{method}")
    plt.show()
    plt.close()


def mse_vs_complexity(x_values, y_values, z_values, max_degree: int = 10, test_size: float = 0.2):
    """
    Function for plotting the MSE for both training and testing data, 
    over complexity. Where complexity means the degree of the polynomal 
    that will define the design matrix

    :param x_values (np.ndarray):
        dependent variable
    :param y_values (np.ndarray):
        dependent variable
    :param z_values (np.ndarray):
        response variable
    :param max_degree (int):
        the maximum order of the polynomial that defines the design matrix
    :param test_size (float)
        the amount of data we will use in testing
    """

    # Spliting in training and testing data
    x_train, x_test, y_train, y_test, z_train, z_test = helper.train_test_split(
        x_values, y_values, z_values, test_size=test_size)

    # Store the MSE values for test- and train data
    list_MSE_testing = []
    list_MSE_training = []

    # Finding MSE for training and test data for different degrees
    for degree in range(1, max_degree + 1):

        # Get the predicted values given our train- and test data
        z_pred_test, z_pred_train, _ = helper.predict_output(
            x_train=x_train, y_train=y_train, z_train=z_train,
            x_test=x_test, y_test=y_test,
            degree=degree, regression_method='OLS',
        )

        # Finding the MSE's
        MSE_test = helper.mean_squared_error(z_test, z_pred_test)
        MSE_train = helper.mean_squared_error(z_train, z_pred_train)

        list_MSE_testing.append(MSE_test)
        list_MSE_training.append(MSE_train)

    plt.plot(range(1, max_degree + 1),
             list_MSE_testing, label="Test sample")
    plt.plot(range(1, max_degree + 1),
             list_MSE_training, label="Training sample")
    plt.xlabel("Model Complexity")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.title(
        f"MSE vs. complexity of the model\nData points: {len(x_values)}; Method: OLS"
    )
    plt.savefig(
        f"figures/MSE_vs_Complexity_DP_{len(x_values)}_d_{degree}_OLS")
    plt.show()
    plt.close()


def main(x_values, y_values, z_values, max_degree: int = 8, test_size: float = 0.2, n_bootstrap: int = 100):
    """
    Doing what we are expecting in exercise 2:
        - Want to plot MSE vs complexity, with both the training and testing data
        - Perform a bias-variance analysis of the Franke function, MSE vs complexity

    :param x_values (np.ndarray):
        dependent variable
    :param y_values (np.ndarray):
        dependent variable
    :param z_values (np.ndarray):
        response variable
    :param max_degree (int):
        the maximum order (complexity) of the polynomial that will define the design matrix
    :param n_bootstrap (int):
        the number of bootstrap iterations
    """

    mse_vs_complexity(
        x_values, y_values, z_values,
        max_degree=max_degree, test_size=test_size)

    bias_variance_boots_looping_degree(
        x_values, y_values, z_values,
        method="OLS", max_degree=max_degree,
        n_bootstrap=n_bootstrap,
        test_size=test_size, show_plot=True)


if __name__ == '__main__':
    # n = 200, noise = 0.0 - 0.1, max_degree = 8 -> great bias_variance
    n = 180
    noise = 0.05
    max_degree = 8
    x_values, y_values, z_values = helper.generate_data(n, noise)
    main(x_values, y_values, z_values, max_degree)
