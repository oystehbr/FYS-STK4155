import helper
import exercise2
import numpy as np


def _get_test_and_train_block(values, i: int, test_no: int, last: bool = False):
    """
    Function that gives out the i-th block of test-data, and gives
    the remaining as training-data

    Usage: for dividing the data in cross-validation

    :param values (np.ndarray):
        the values you wanna get the i-th block of data
    :param i (int):
        indicate the block_no, we want to get
    :param test_no (int):
        amount of test data
    :param last (bool):
        - True: the last block of testing data
        - False: not the last block of testing data

    :return (tuple[np.ndarray, np.ndarray]):
        train and test data
    """

    if not last:
        train = np.concatenate((
            values[0:i*test_no],
            values[(i+1)*test_no:]
        ))
        test = values[i*test_no:(i+1)*test_no]
    else:
        train = values[0:i*test_no]
        test = values[i*test_no:]

    return train, test


def cross_validation(x_values, y_values, z_values, method: str, k_folds: int = 5, degree: int = 5, lmbda: float = 1):
    """
    Function for finding the MSE with usage of the 
    resampling technique: cross-validation

    :param x_values (np.ndarray):
        dependent variable
    :param y_values (np.ndarray):
        dependent variable
    :param z_values (np.ndarray):
        response variable
    :param method (str):
        the preffered regression method: OLS, RIDGE or LASSO
    :param k_folds (int):
        the amount of folds we wanna do cross-validation on
    :param degree (int):
        the order of the polynomial that defines the design matrix
    :param lmbda (float):
        parameter used by Ridge and Lasso regression (lambda)

    :return (None, tuple[list, list, list]):
        - MSE for the given model
        - bias for the given model
        - variance for the given model 
    """

    # Find the correct number of test data
    if len(x_values) % k_folds > 2:
        test_no = len(x_values) // k_folds + 1
    else:
        test_no = len(x_values) // k_folds

    last = False
    MSE_list = np.zeros(k_folds)
    for i in range(k_folds):
        if i == (k_folds - 1):
            # If last "block" -> get the rest of the data
            last = True

        x_train, x_test = _get_test_and_train_block(x_values, i, test_no, last)
        y_train, y_test = _get_test_and_train_block(y_values, i, test_no, last)
        z_train, z_test = _get_test_and_train_block(z_values, i, test_no, last)

        # Get the predicted values given our train- and test data
        z_pred_test, _, _ = helper.predict_output(
            x_train=x_train, y_train=y_train, z_train=z_train,
            x_test=x_test, y_test=y_test,
            degree=degree, regression_method=method,
            lmbda=lmbda
        )

        MSE_list[i] = helper.mean_squared_error(z_test, z_pred_test)

    estimated_MSE_cross_validation = np.mean(MSE_list)

    # Get MSE from bootstrap for comparison
    estimated_MSE_bootstrap = exercise2.bias_variance_boots_looping_degree(
        x_values, y_values, z_values,
        method, max_degree=degree,
        lmbda=lmbda)[0][-1]

    return estimated_MSE_cross_validation, estimated_MSE_bootstrap


def main(x_values, y_values, z_values, degree: int, k_folds: int = 5):
    """
    Doing what we are expecting in exercise 3:
        - Compare the MSE you get from cross-validation with the one from bootstrap

    :param x_values (np.ndarray):
        dependent variable
    :param y_values (np.ndarray):
        dependent variable
    :param z_values (np.ndarray):
        response variable
    :param degree (int):
        the order of the polynomial that will define the design matrix
    """

    MSE_cross_validation, MSE_bootstrap = cross_validation(
        x_values, y_values, z_values,
        method='OLS',  k_folds=k_folds, degree=degree)

    print(f'MSE_cross: {MSE_cross_validation}')
    print(f'MSE_boot: {MSE_bootstrap}')


if __name__ == "__main__":
    n = 100
    noise = 0.1
    degree = 8
    x_values, y_values, z_values = helper.generate_data(n, noise)
    main(x_values, y_values, z_values, degree)
