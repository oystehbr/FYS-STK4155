import exercise1
import exercise2
import exercise3
import exercise4
import numpy as np
import helper
import matplotlib.pyplot as plt


def get_betas_RIDGE(X, lmbda, z_values):
    """
    TODO: docstrings

    """
    # TODO: make i RIDGE
    X_T = np.matrix.transpose(X)
    p = X.shape[1]
    I = np.eye(p, p)

    betas = np.linalg.pinv(X_T @ X + lmbda*I) @ X_T @ z_values

    return betas


# Perform the same bootstrap analysis as in Exercise 2
# TODO: (for the same polynomials)
def ridge_bootstrap_analysis_vs_complexity(x_values, y_values, z_values, max_degree=12, test_size=0.2, lmbda=0.1):
    """
    FIRST TASK
    # TODO: docstrings

    """
    exercise2.bias_variance_boots(x_values=x_values,
                                  y_values=y_values,
                                  z_values=z_values,
                                  method='RIDGE',
                                  min_degree=1,
                                  max_degree=max_degree,
                                  test_size=test_size,
                                  show_plot=True,
                                  lmbda=lmbda)


def ridge_bootstrap_analysis_vs_lmbda(x_values, y_values, z_values, degree, test_size=0.2, lmbda=0.1):
    return exercise2.bias_variance_boots(x_values=x_values,
                                         y_values=y_values,
                                         z_values=z_values,
                                         method='RIDGE',
                                         min_degree=degree,
                                         max_degree=degree,
                                         test_size=test_size,
                                         show_plot=False,
                                         lmbda=lmbda)


def ridge_cross_validation(x_values, y_values, z_values, degree=12, lmbda=0.1):
    """
    # TODO: docstrings

    """

    return_values = exercise3.cross_validation(x_values=x_values,
                                               y_values=y_values,
                                               z_values=z_values,
                                               method='RIDGE',
                                               k_folds=5,
                                               degree=degree,
                                               lmbda=lmbda)
    cross_MSE_estimate, boots_MSE_estimate = return_values
    return cross_MSE_estimate, boots_MSE_estimate


def main(x_values, y_values, z_values, max_degree, degree):

    n = len(x_values)
    # bias-variance vs complexity
    ridge_bootstrap_analysis_vs_complexity(x_values, y_values, z_values,
                                           max_degree=max_degree, test_size=0.2, lmbda=0.1)

    # Cross-validation with ridge
    number_of_lambdas = 40
    lmbdas = np.logspace(-4, 4, number_of_lambdas)
    cross_MSE_estimates = []
    for lmbda in lmbdas:
        cross_MSE_estimate, _ = ridge_cross_validation(x_values, y_values, z_values,
                                                       degree=degree, lmbda=lmbda)
        cross_MSE_estimates.append(cross_MSE_estimate)

    # TODO: nicer plots
    plt.plot(np.log10(lmbdas), cross_MSE_estimates)
    plt.show()

    # bias-variance vs parameter lambda
    mse_list = []
    bias_list = []
    variance_list = []

    # TODO: better explanation
    for lmbda in lmbdas:
        results = ridge_bootstrap_analysis_vs_lmbda(
            x_values, y_values, z_values, degree=degree, lmbda=lmbda)
        mse, bias, variance = results
        mse_list.append(mse[-1])
        bias_list.append(bias[-1])
        variance_list.append(variance[-1])

    plt.plot(np.log10(lmbdas), mse_list, label="MSE")
    plt.plot(np.log10(lmbdas), bias_list, label="BAIS")
    plt.plot(np.log10(lmbdas), variance_list, label="VARIANCE")
    plt.xlabel(f"log_10(lambda)")
    plt.ylabel(f"Error")
    plt.legend()
    plt.title(f"Error vs lambda\nData points: {n}\nDegree: {degree}")
    plt.show()


if __name__ == "__main__":
    n = 200
    noise = 0.2
    max_degree = 18
    degree = 5
    x_values, y_values, z_values = helper.generate_data(n, noise)
    main(x_values, y_values, z_values, max_degree, degree)
