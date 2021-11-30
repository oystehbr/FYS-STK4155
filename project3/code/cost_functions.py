import autograd.numpy as np
import numpy
from activation_functions import softmax
import helper


def logistic_cost_NN(y_hat, y):
    """
    Finding the cost w.r.t. the logistic cost function,
    used to determine the cost in a classification case of the Neural Network
    with binary target.

    :param y_hat (np.ndarray): 
        predicted target
    :param y (np.ndarray): 
        actual target

    :return (number):
        the cost value
    """

    return -np.sum(y*np.log(y_hat) + (1-y)*np.log(1 - y_hat))


def logistic_cost_NN_multi(y_hat, y):
    """
    Finding the cost w.r.t. the logistic cost function,
    used to determine the cost in a classification case of the Neural Network
    with several targets.
    
    Link: 
    https://www.analyticsvidhya.com/blog/2021/02/cost-function-is-no-rocket-science/

    :param y_hat (np.ndarray): 
        predicted target
    :param y (np.ndarray): 
        actual target

    :return (number):
        the cost value
    """

    # Row-wise dot product, then summation
    return - np.sum(np.sum(np.log(y_hat)*y, axis=1))


def cost_logistic_regression(beta, X, y, lmbda=0):
    """
    The cost function of the logistic regression, calculates the 
    predicted values w.r.t. the given betas and X.

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

    return -np.sum(y*np.log(prob(beta, X)) + (1-y)
                   * np.log(1 - prob(beta, X))) + lmbda*np.sum(beta**2)


def cost_logistic_regression_multi(beta, X, y, lmbda=0):
    """
    The cost function of the logistic regression, calculates the 
    predicted values w.r.t. the given betas and X.

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

    # Row-wise dot product, then summation
    return - np.sum(np.sum(np.log(prob_multi(beta, X))*y, axis=1)) + lmbda*np.sum(beta**2)


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

    return (np.exp(beta[0] + X @ beta[1:]) / (1 + np.exp(beta[0] + X @ beta[1:]))).reshape(-1, 1)


def prob_multi(beta, X):
    """
    helper function for cost_logistic_regression_multi, will
    establish the probabilities

    :param beta (np.ndarray):
        input value
    :param x (np.ndarray, number):
        input value

    :return (np.ndarray):
        function value
    """

    z = beta[:, 0].T + X @ beta[:, 1:].T

    probs = softmax(z)

    return probs


def MSE(y_hat, y):
    """
    Sort of a MSE function, with another scalar

    :param y_hat (np.ndarray):
        the predicted target
    :param y (np.ndarray): 
        the target

    :return (number):
        the value
    """

    return 1/len(y_hat) * np.sum((y - y_hat)**2)


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
