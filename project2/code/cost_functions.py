import autograd.numpy as np


def logistic_cost(y_hat, y):
    """
    Finding the cost w.r.t. the logistic cost function,
    used to determine the cost in the Neural Network

    :param y_hat (np.ndarray): 
        predicted target
    :param y (np.ndarray): 
        actual target

    :return (number):
        the cost value
    """

    return -np.sum(y*np.log(y_hat) + (1-y)*np.log(1 - y_hat))

    # TODO: delete underneath
    sum = 0
    m = y.shape[0]
    # for [y_i], [y_hat_i] in zip(y, y_hat):
    #     sum += y_i * np.log(y_hat_i) + (1-y_i) * np.log(1 - y_hat_i)

    # print(abs(sum1+sum))
    return -sum


def cost_logistic_regression(beta, X, y, lmbda=0):
    """
    The cost function of the logistic regression, with given betas
    # TODO: can it merge into one method?

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
    # TODO: send in the fully X?
    n = X.shape[0]
    total_prob = prob(beta, X[0])**y[0]*(1-prob(beta, X[0]))**(1-y[0])

    for i in range(1, n):
        p = prob(beta, X[i])
        total_prob *= p**y[i]*(1-p)**(1 - y[i])

    # TODO: solve this problem
    total_prob1 = prob(beta, X)**y*(1-prob(beta, X))**(1-y)
    print(total_prob)
    print(np.prod(total_prob1))
    print('STOP')
    return - np.log(total_prob) + lmbda*np.sum(beta**2)


def prob(beta, X):
    """
    helper function for cost_logistic_regression, will
    establish the probability:
        P (y=1 | x, beta)

    :param beta (np.ndarray):
        input value
    :param x (np.ndarray, number):
        input value

    :return (np.ndarray, number):
        function value
    """
    # TODO: delete this
    # np.sum([_beta*x**i for i, _beta in enumerate(beta)])
    # print(np.sum([_beta*x**i for i, _beta in enumerate(beta)]))
    # print(beta[0] + beta[1]*x)

    return np.exp(X @ beta) / (1 + np.exp(X @ beta))


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
