from numba.cuda import test
import exercise1
import exercise2
import exercise3
import exercise4
import numpy as np
import helper
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


def lasso_bootstrap_analysis_vs_complexity(x_values, y_values, z_values, max_degree: int, n_bootstrap: int = 100, test_size: float = 0.2, lmbda: float = 1):
    """
    Function for plotting MSE, bias and variance (testing data),
    over complexity. Where complexity means the order of the polynomial
    that defines the design matrix. Regression method: Lasso

    :param x_values (np.ndarray):
        dependent variable
    :param y_values (np.ndarray):
        dependent variable
    :param z_values (np.ndarray):
        response variable
    :param max_degree (int):
        the maximum order of the polynomial that defines the design matrix
    :param n_bootstrap (int):
        the number of bootstrap iterations
    :param test_size (float)
        the amount of data we will use in testing
    :param lmbda (float):
        parameter used in the regression method
    """

    exercise2.bias_variance_boots_looping_degree(
        x_values=x_values, y_values=y_values, z_values=z_values,
        method='LASSO', max_degree=max_degree,
        n_bootstrap=n_bootstrap, test_size=test_size,
        show_plot=True, lmbda=lmbda)


def lasso_bootstrap_analysis_vs_lmbda(x_values, y_values, z_values, degree: int, n_bootstrap: int = 100, test_size=0.2):
    """
    Function for plotting MSE, bias and variance (testing data),
    over lambda. It will be using Lasso regression. 

    :param x_values (np.ndarray):
        dependent variable
    :param y_values (np.ndarray):
        dependent variable
    :param z_values (np.ndarray):
        response variable
    :param degree (int):
        the order of the polynomial that defines the design matrix
    :param n_bootstrap (int):
        the number of bootstrap iterations
    :param test_size (float)
        the amount of data we will use in testing
    """

    # MSE, bias and variance lists
    mse_list = []
    bias_list = []
    variance_list = []

    number_of_lambdas = 20
    lmbdas = np.logspace(-4, 4, number_of_lambdas)

    # For every lambda, get MSE, bias and variance
    for lmbda in lmbdas:
        mse, bias, variance = exercise2.bias_variance_boots_looping_lambda(
            x_values=x_values, y_values=y_values, z_values=z_values,
            method='LASSO', degree=degree, n_bootstrap=n_bootstrap,
            test_size=test_size, lmbda=lmbda)

        mse_list.append(mse[-1])
        bias_list.append(bias[-1])
        variance_list.append(variance[-1])

    plt.plot(np.log10(lmbdas), mse_list, label="MSE")
    plt.plot(np.log10(lmbdas), bias_list, label="BAIS")
    plt.plot(np.log10(lmbdas), variance_list, label="VARIANCE")
    plt.title(
        f"Error vs lambda\nData points: {len(x_values)}; Degree: {degree}; Method: LASSO")
    plt.xlabel("log10(lmbdas)")
    plt.ylabel("Prediction error")
    plt.legend()
    plt.savefig(
        f"figures/mse_boots_looping_lambda_DP_{len(x_values)}_d_{degree}_LASSO")
    plt.show()


def lasso_cross_validation(x_values, y_values, z_values, degree: int = 12, k_folds: int = 5):
    """
    The function will calculate the MSE with the sampling method:
    cross-validation. It will use Lasso regression and will plot the 
    MSE vs lmbda. 

    :param x_values (np.ndarray):
        dependent variable
    :param y_values (np.ndarray):
        dependent variable
    :param z_values (np.ndarray):
        response variable
    :param degree (int):
        the order of the polynomial that will define the design matrix
    :param k_folds (int):
        the amount of folds we wanna do cross-validation on
    """

    number_of_lambdas = 80
    lmbdas = np.logspace(-3, 2, number_of_lambdas)
    cross_MSE_estimates = []

    # For every lambda, get MSE with cross-validation
    for lmbda in lmbdas:
        cross_MSE_estimate, _ = exercise3.cross_validation(
            x_values=x_values, y_values=y_values, z_values=z_values,
            method='LASSO', k_folds=k_folds, degree=degree,
            lmbda=lmbda)

        cross_MSE_estimates.append(cross_MSE_estimate)

    plt.plot(np.log10(lmbdas), cross_MSE_estimates)
    plt.title(
        f"MSE vs lambda\nData points: {len(x_values)}; Degree: {degree}; Method: LASSO")
    plt.ylabel("MSE")
    plt.xlabel("log10(lmbdas)")
    plt.savefig(
        f"figures/mse_cross_validation_vs_lambda_DP_{len(x_values)}_d_{degree}_LASSO")
    plt.show()


def main(x_values, y_values, z_values, max_degree: int, degree: int, test_size: float = 0.2, k_folds: int = 5,  n_bootstrap: int = 100, lmbda: float = 0.1):
    """
    Doing what we are expecting in exercise 5:
        - Perform the same bootstrap analysis (mse vs complexity) as in exercise 2
        - Perform cross-validation as in exercise 3 (different values of lmbda)
        - Perform a bias-variance analysis (MSE vs lmbda)

    :param x_values (np.ndarray):
        dependent variable
    :param y_values (np.ndarray):
        dependent variable
    :param z_values (np.ndarray):
        response variable
    :param max_degree (int):
        the maximum order of the polynomial that will define the design matrix
    :param degree (int):
        the order of the polynomial that will define the design matrix
    :param test_size (float):   
        the amount of data we will use in testing
    :param k_folds (int):
        the amount of folds we wanna do cross-validation on
    :param n_bootstrap (int):
        the number of bootstrap iterations
    :param lmbda (float):
        parameter used in the regression method
    """

    # Perform bootstrap analysis for Lasso
    lasso_bootstrap_analysis_vs_complexity(
        x_values, y_values, z_values,
        max_degree=max_degree, n_bootstrap=n_bootstrap,
        test_size=test_size, lmbda=lmbda)

    # Perform cross-validation with Lasso
    lasso_cross_validation(
        x_values, y_values, z_values,
        degree=degree, k_folds=k_folds)

    # Bias-variance trade-off vs parameter lambda
    lasso_bootstrap_analysis_vs_lmbda(
        x_values, y_values, z_values,
        degree=degree, n_bootstrap=n_bootstrap,
        test_size=test_size)


if __name__ == "__main__":
    n = 119
    noise = 0.1
    max_degree = 20
    degree = 5
    x_values, y_values, z_values = helper.generate_data(n, noise)

    main(x_values, y_values, z_values, max_degree, degree, lmbda=0.001)
