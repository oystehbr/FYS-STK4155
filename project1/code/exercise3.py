import helper
import exercise2
import numpy as np

# TODO: Make this even better


def _get_test_and_train_block(values: np.ndarray, i: int, test_no: int, last: bool = False):
    """
    Function that gives out the i-th block of test-data, and gives
    the remaining as training-data

    Usage: for dividing the data in cross-validation

    :param values (np.ndarray):
        the values you wanna get blocks of data of
    :param i (int):
        indicate the block_no, we want to get
    :param test_no (int):
        amount of test data
    # TODO: bool

    :return (tuple):
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


def cross_validation(x_values, y_values, z_values, method, k_folds=5, degree=5, lmbda=1):
    """
    # TODO: docs

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


def main(x_values, y_values, z_values, degree):

    MSE_cross_validation, MSE_bootstrap = cross_validation(x_values, y_values, z_values,
                                                           method='OLS',  k_folds=5, degree=degree)
    print(f'MSE_cross: {MSE_cross_validation}')
    print(f'MSE_boot: {MSE_bootstrap}')


if __name__ == "__main__":
    n = 100
    noise = 0.1
    degree = 8
    x_values, y_values, z_values = helper.generate_data(n, noise)
    main(x_values, y_values, z_values, degree)
