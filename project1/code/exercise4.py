import exercise1
import exercise2
import exercise3
import exercise4
import numpy as np
import helper
import matplotlib.pyplot as plt


# Perform the same bootstrap analysis as in Exercise 2
# TODO: (for the same polynomials)
def ridge_bootstrap_analysis_vs_complexity(x_values, y_values, z_values, max_degree=12, test_size=0.2, lmbda=0.1):
    """
    FIRST TASK
    # TODO: docstrings

    """
    exercise2.bias_variance_boots_looping_degree(
        x_values=x_values, y_values=y_values, z_values=z_values,
        method='RIDGE', max_degree=max_degree,
        test_size=test_size, show_plot=True,
        lmbda=lmbda)


def ridge_bootstrap_analysis_vs_lmbda(x_values, y_values, z_values, degree, test_size=0.2):
    """
    # TODO: docst
    """

    # bias-variance vs parameter lambda
    mse_list = []
    bias_list = []
    variance_list = []

    # TODO: switch input according to what Morten says
    number_of_lambdas = 40
    lmbdas = np.logspace(-4, 4, number_of_lambdas)
    # TODO: better explanation
    for lmbda in lmbdas:
        results = exercise2.bias_variance_boots_looping_lambda(
            x_values=x_values, y_values=y_values, z_values=z_values,
            method='RIDGE', degree=degree,
            test_size=test_size,
            lmbda=lmbda)

        mse, bias, variance = results
        mse_list.append(mse[-1])
        bias_list.append(bias[-1])
        variance_list.append(variance[-1])

    plt.plot(np.log10(lmbdas), mse_list, label="MSE")
    plt.plot(np.log10(lmbdas), bias_list, label="BAIS")
    plt.plot(np.log10(lmbdas), variance_list, label="VARIANCE")
    plt.xlabel(f"log_10(lambda)")
    plt.ylabel(f"Error")
    plt.title(
        f"Error vs lambda\nData points: {len(x_values)}; Degree: {degree}; Method: RIDGE")
    plt.legend()
    plt.show()


def ridge_cross_validation(x_values, y_values, z_values, degree=12, lmbda=0.1):
    """
    # TODO: docstrings

    """
    # TODO: put something in signature -> according to Morten
    number_of_lambdas = 40
    lmbdas = np.logspace(-4, 4, number_of_lambdas)
    cross_MSE_estimates = []
    for lmbda in lmbdas:
        cross_MSE_estimate, _ = exercise3.cross_validation(
            x_values=x_values, y_values=y_values, z_values=z_values,
            method='RIDGE', k_folds=5,
            degree=degree, lmbda=lmbda)

        cross_MSE_estimates.append(cross_MSE_estimate)

    # TODO: nicer plots
    plt.plot(np.log10(lmbdas), cross_MSE_estimates)
    plt.title(
        f"MSE vs lambda\nData points: {len(x_values)}; Degree: {degree}; Method: RIDGE")
    plt.ylabel("MSE")
    plt.xlabel("log10(lmbdas)")
    plt.show()


def main(x_values, y_values, z_values, max_degree, degree):

    # Perform bootstrap analysis for Ridge
    ridge_bootstrap_analysis_vs_complexity(
        x_values, y_values, z_values,
        max_degree=max_degree, test_size=0.2, lmbda=0.1)

    # Perform cross-validation with Ridge
    ridge_cross_validation(
        x_values, y_values, z_values,
        degree=degree)

    # Bias-variance trade-off vs parameter lambda
    ridge_bootstrap_analysis_vs_lmbda(
        x_values, y_values, z_values,
        degree=degree)


if __name__ == "__main__":
    n = 200
    noise = 0.2
    max_degree = 10
    degree = 5
    x_values, y_values, z_values = helper.generate_data(n, noise)
    main(x_values, y_values, z_values, max_degree, degree)
