import autograd.numpy as np


def logistic_cost(y_hat, y):
    sum = 0
    m = y.shape[0]

    return -np.sum(y*np.log(y_hat) + (1-y)*np.log(1 - y_hat))
    # for [y_i], [y_hat_i] in zip(y, y_hat):
    #     sum += y_i * np.log(y_hat_i) + (1-y_i) * np.log(1 - y_hat_i)

    # print(abs(sum1+sum))
    return -sum


def MSE(y_hat, y):
    return 1/2 * np.sum((y - y_hat)**2)


def cost_OLS(beta, X, y, lmbda=0):
    """
    The cost function of the regression method OLS

    :param beta (np.ndarray):
        the regression parameters
    :param X (np.ndarray):
        input values (dependent variables)
    :param y (np.ndarray):
        actual output values
    :param lmbda (float):
        do not think about this, it will not be used. Just for simplicity of
        the code structure of the SGD

    :return (float):
        the value of the cost function
    """

    # Find the predicted values according to the given betas and input values
    y_pred = X @ beta

    return np.mean((y_pred - y)**2)


def cost_RIDGE(beta, X, y, lmbda):
    """
    The cost function of the regression method RIDGE

    :param beta (np.ndarray):
        the regression parameters
    :param X (np.ndarray):
        input values (dependent variables)
    :param y (np.ndarray):
        actual output values
    :param lmbda (float):
        the hyperparameter for RIDGE regression

    :return (float):
        the value of the cost function
    """

    # Find the predicted values according to the given betas and input values
    y_pred = X @ beta

    return np.mean((y_pred - y)**2) + lmbda*np.sum(beta**2)
