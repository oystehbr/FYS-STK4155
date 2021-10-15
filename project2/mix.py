import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad as egrad


def gradient_descent(theta_init, eta, C, N, x, y, gamma=0, tol=1e-14):
    """
    This function will get a better theta value (from the initial
    guess) for optimizing the given cost-function. Returning the 
    new theta values. 


    :param theta_init (np.ndarray):
        the initial guess of our parameters
    :param eta (float): 
        the learning rate for our gradient_descent method
    :param C (function):
        the cost-function we will optimize
    :param N (int):
        number of maximum iterations
    :param tol (float):
        stop the iteration if the distance is less than this tolerance
    :param gamma (float):
        momentum parameter between 0 and 1, (0 if no momentum)

    :return tuple(np.ndarray, int):
        - better estimate of theta, according to the cost-function
        - number of iterations, before returning
    """

    grad_C = egrad(C)
    theta_previous = theta_init

    n = len(x)
    M = 10  # size of minibatch
    m = int(n/M)

    # If we want the general gradient decent -> two functions in class tho
    # TODO: maybe update eta value
    v = eta*grad_C(theta_init, x, y)

    # TODO: updating v, will be wrong if no GD, make class Brolmsen
    for i in range(N):
        grad = grad_C(theta_previous, x, y)
        # Momentum based GD
        v = gamma*v + eta*grad
        theta_next = theta_previous - v

        # If the iterations are not getting any better
        if np.sum(np.abs(theta_next - theta_previous)) < tol:
            return theta_next, i

        # Updating the thetas
        theta_previous = theta_next

    return theta_next, N
