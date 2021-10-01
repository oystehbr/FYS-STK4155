from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import helper
import numpy as np


def find_variance(z_true, z_pred):
    """
    # TODO: docs
    """

    n = len(z_true)
    sum = 0
    for t, p in zip(z_true, z_pred):
        sum += (t - p)**2
    return sum/n


def get_confidence_interval_ND(betas, X, z_true, CI_num=0.95):
    """
    Function that calculate the confidence interval for the given bates
    returns the interval as a list

    # TODO: docstrings

    """

    # TODO: sigma_squared found right
    z_pred = X @ betas
    sigma_squared = find_variance(z_true, z_pred)

    X_T = np.matrix.transpose(X)
    cov_matrix = sigma_squared * np.linalg.inv(X_T @ X)

    # TODO: calculate z, w.r.t CI_num
    z = 1.96

    n = len(z_true)
    list_of_confidence_intervals = []
    # TODO: explain what happens here
    for i, mean_beta in enumerate(betas):
        sigma_beta = np.sqrt(sigma_squared * cov_matrix[i][i])
        CI = [mean_beta - z*sigma_beta,
              mean_beta + z*sigma_beta]
        list_of_confidence_intervals.append(CI)

    return list_of_confidence_intervals


def print_betas_CI(betas, CI_list):
    """
    # TODO: docsstrings
    """

    print(f'Beta (No)  Beta-value  CI')
    for i, (beta, CI) in enumerate(zip(betas, CI_list)):
        print(f'{i} | {beta} | {CI}')


def main(x_values, y_values, z_values, degree=5, test_size=0.2):
    """
    # TODO: docstrings

    """

    # We split the data in test and training data
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        x_values, y_values, z_values, test_size=test_size)

    # TODO: same comment on every such case
    z_pred_test, _, betas_OLS = helper.predict_output(
        x_train=x_train, y_train=y_train, z_train=z_train,
        x_test=x_test, y_test=y_test,
        degree=degree, regression_method='OLS'
    )

    # Find the confidence intervals of the betas # TODO: confidence interval scaled
    X_train = helper.create_design_matrix(x_train, y_train, degree)
    CI_list = get_confidence_interval_ND(
        betas_OLS, X_train, z_train)

    print_betas_CI(betas_OLS, CI_list)

    # Evaluating the Mean Squared error (MSE) and R2-score
    MSE = helper.mean_squared_error(z_test, z_pred_test)
    R2_score = helper.r2_score(z_test, z_pred_test)

    print(f'MSE: {MSE}')
    print(f'R2_score: {R2_score}')


if __name__ == "__main__":
    n = 1000
    noise = 0
    x_values, y_values, z_values = helper.generate_data(n, noise)
    main(x_values, y_values, z_values)
