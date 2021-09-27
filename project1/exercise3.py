import numpy as np
import exercise1


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


def cross_validation(x_values, y_values, z_values, k_folds=5):
    """
    # TODO: docs

    """
    # TODO: x_values and y_values together? INTO X, and better normalizing data

    test_no = len(x_values) // k_folds

    last = False
    for i in range(k_folds):
        if i == (k_folds - 1):
            last = True

        x_train, x_test = _get_test_and_train_block(x_values, i, test_no, last)
        y_train, y_test = _get_test_and_train_block(x_values, i, test_no, last)
        z_train, z_test = _get_test_and_train_block(x_values, i, test_no, last)

        if False:
            # Test for checking that it gives the correct output
            print(f'ROUND {i}')
            print(x_values)
            print(f'x_train: {x_train}, len: {len(x_train)} ')
            print(f'x_test: {x_test}, len: {len(x_test)} ')
            print('------------------------')

        # TODO: what do we want with this

    return
    # z_pred_test_matrix = np.empty((z_test.shape[0], n_bootstrap))

    # Running bootstrap-method on training data -> collect the different betas
    for i in range(n_bootstrap):
        _x, _y, _z = resample(x_train, y_train, z_train)

        # Evaluate the new model on the same test data each time.
        betas, _ = exercise1.get_betas_and_design_matrix(
            _x, _y, _z, degree)
        z_pred_test_matrix[:, i] = exercise1.z_predicted(X_test, betas)

    # Take the mean of the betas from the bootstrap
    z_pred_test = np.mean(z_pred_test_matrix, axis=1, keepdims=True)

    pass


def main():
    n = 10
    x_values, y_values, z_values = exercise1.generate_data(n, 0)
    cross_validation(x_values, y_values, z_values)

    # # Scale data before further use
    # TODO: do we scale correct??
    # x_values, y_values, z_values = exercise1.scaling_the_data(
    #     x_values, y_values, z_values)

    pass


if __name__ == "__main__":
    main()
