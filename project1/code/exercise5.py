import exercise1
import exercise2
import exercise3
import exercise4
import numpy as np
import helper
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso


def lasso_bootstrap_analysis_vs_complexity(x_values, y_values, z_values, max_degree=12, test_size=0.2, lmbda=0.1):
    """
    FIRST TASK
    # TODO: docstrings

    """
    exercise2.bias_variance_boots_looping_degree(
        x_values=x_values, y_values=y_values, z_values=z_values,
        method='LASSO', max_degree=max_degree,
        test_size=test_size, show_plot=True,
        lmbda=lmbda)


def lasso_bootstrap_analysis_vs_lmbda(x_values, y_values, z_values, degree, test_size=0.2):
    """
    TODO: docs
    """
    # TODO: morten
    # bias-variance vs parameter lambda
    mse_list = []
    bias_list = []
    variance_list = []

    number_of_lambdas = 100
    lmbdas = np.logspace(-2, 2, number_of_lambdas)

    for lmbda in lmbdas:
        mse, bias, variance = exercise2.bias_variance_boots_looping_lambda(
            x_values=x_values, y_values=y_values, z_values=z_values,
            method='LASSO', degree=degree, test_size=test_size, lmbda=lmbda)

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
    plt.show()


def lasso_cross_validation(x_values, y_values, z_values, degree=12):
    """
    # TODO: docstrings

    """
    # TODO: MORTEN ANSWER
    number_of_lambdas = 100
    # TODO: start_value -2 because of convergenceWarning
    lmbdas = np.logspace(-2, 2, number_of_lambdas)
    cross_MSE_estimates = []
    for lmbda in lmbdas:
        cross_MSE_estimate, _ = exercise3.cross_validation(
            x_values=x_values, y_values=y_values, z_values=z_values,
            method='LASSO', k_folds=5, degree=degree,
            lmbda=lmbda)

        cross_MSE_estimates.append(cross_MSE_estimate)

    # TODO: nicer plots
    plt.plot(np.log10(lmbdas), cross_MSE_estimates)
    plt.title(
        f"MSE vs lambda\nData points: {len(x_values)}; Degree: {degree}; Method: LASSO")
    plt.ylabel("MSE")
    plt.xlabel("log10(lmbdas)")
    plt.show()


def main(x_values, y_values, z_values, max_degree, degree):

    # Perform bootstrap analysis for Lasso
    lasso_bootstrap_analysis_vs_complexity(x_values, y_values, z_values,
                                           max_degree=max_degree, test_size=0.2, lmbda=0.001)

    # Perform cross-validation with Lasso
    lasso_cross_validation(x_values, y_values, z_values,
                           degree=degree)

    # Bias-variance trade-off vs parameter lambda
    lasso_bootstrap_analysis_vs_lmbda(
        x_values, y_values, z_values,
        degree=degree)


if __name__ == "__main__":
    n = 200
    noise = 0.2
    max_degree = 10
    degree = 5
    x_values, y_values, z_values = helper.generate_data(n, noise)

    main(x_values, y_values, z_values, max_degree, degree)
