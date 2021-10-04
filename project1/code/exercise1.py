import helper
import numpy as np


def get_confidence_interval_beta(beta, X, z_true):
    """
    Function that calculate the 95% confidence interval for the given betas
    returns the intervals as a list

    :param beta (np.ndarray):
        the regression parameters (scaled version)
    :param X (np.ndarray):
        design matrix
    :param z_true (np.ndarray):
        the true response variables

    :return (list):
        with 95% condfidence intervals for beta, in chronological order
    """

    # Want to predict z, betas is scaled -> so we need to scale X
    X_scaled = X - np.mean(X, axis=0)
    z_pred = X_scaled @ beta + np.mean(z_true, axis=0)  # scaling it back

    # Estimating sigma_squared with MSE
    sigma_squared = helper.mean_squared_error(z_true, z_pred)

    X_T = np.matrix.transpose(X)
    cov_matrix = sigma_squared * np.linalg.inv(X_T @ X)

    # Chosen for finding 95% CI
    z = 1.96

    # Calculating the different confidence intervals
    list_of_confidence_intervals = []
    for i, mean_beta in enumerate(beta):
        sigma_beta = np.sqrt(sigma_squared * cov_matrix[i][i])
        CI = [mean_beta - z*sigma_beta,
              mean_beta + z*sigma_beta]
        list_of_confidence_intervals.append(CI)

    return list_of_confidence_intervals


def print_beta_CI(beta, CI_list):
    """
    Function that takes the regression parameters and their 
    corresponding confidence interval, and prints out a 
    nicely formatted table 

    :param beta (np.ndarray):
        the regression parameters
    :param CI_list (list):
        the confidence interval for the regression parameters 
    """

    print(f'Beta (No) | Beta-value |         95% CI')
    for i, (beta, CI) in enumerate(zip(beta, CI_list)):
        print(f'{i: 9.0f} | {beta: 10.3f} | [{CI[0]: 8.3f}, {CI[1]: 7.3f}]')


def main(x_values, y_values, z_values, degree: int = 5, test_size: float = 0.2):
    """
    Doing what we are expecting in exercise 1:
        - Performing an OLS analysis using polynomials in x and y (_values in our case)
        - Find the confidence intervals of the parameters beta (printing them out in nice format)
        - Evaluating the Mean Squared error (MSE)
        - Evaluating the R^2 score

    :param x_values (np.ndarray):
        dependent variable
    :param y_values (np.ndarray):
        dependent variable
    :param z_values (np.ndarray):
        response variable
    :param degree (int):
        the order of the polynomial that will define the design matrix
    :param test_size (float)
        the amount of data we will use in testing
    """

    # We split the data in test and training data
    x_train, x_test, y_train, y_test, z_train, z_test = helper.train_test_split(
        x_values, y_values, z_values, test_size=test_size)

    # Predicting z with testing data (given the model of the training data)
    z_pred_test, _, beta_OLS = helper.predict_output(
        x_train=x_train, y_train=y_train, z_train=z_train,
        x_test=x_test, y_test=y_test,
        degree=degree, regression_method='OLS'
    )

    # Finding the 95% confidence intervals for beta
    X_train = helper.create_design_matrix(x_train, y_train, degree)
    CI_list = get_confidence_interval_beta(
        beta_OLS, X_train, z_train)

    # Prining out the confidence interval in a nice way
    print_beta_CI(beta_OLS, CI_list)

    # Evaluating the Mean Squared error (MSE) and R2-score
    MSE = helper.mean_squared_error(z_test, z_pred_test)
    R2_score = helper.r2_score(z_test, z_pred_test)

    print(f'MSE: {MSE}')
    print(f'R2_score: {R2_score}')


if __name__ == "__main__":
    n = 50000
    noise = 0.1
    x_values, y_values, z_values = helper.generate_data(n, noise)
    main(x_values, y_values, z_values)
