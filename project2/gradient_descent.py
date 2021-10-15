import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import elementwise_grad as egrad
from project1 import helper


def SGD(theta_init, eta, C, n_epochs, X, y, gamma=0, tol=1e-14):
    """
    Stochastic Gradient Descent
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

    n = X.shape[0]
    M = 10  # size of minibatch
    m = int(n/M)

    # If we want the general gradient decent -> two functions in class tho
    # TODO: maybe update eta value
    v = eta*grad_C(theta_init, X, y)
    theta_previous = theta_init

    # TODO: updating v, will be wrong if no GD, make class Brolmsen
    j = 0
    for epoch in range(n_epochs):
        for i in range(m):
            # Do something with the end interval of the selected
            k = np.random.randint(m)

            # Finding the k-th batch
            xk_batch = X[k*M:(k+1)*M]
            yk_batch = y[k*M:(k+1)*M]

            grad = grad_C(theta_previous, xk_batch, yk_batch)
            v = gamma*v + eta*grad
            theta_next = theta_previous - v

            j += 1
            # Check if we have reached the tolerance
            if np.sum(np.abs(theta_next - theta_previous)) < tol:
                print('local')
                return theta_next, j

            # Updating the thetas
            theta_previous = theta_next
            print('---start---')
            print(theta_previous)
            print('---stop---')

    return theta_next, j




def cost(beta, X, y):
    """
    cost-function OLS
    # TODO:

    """

    y_pred = X @ beta

    # TODO: make better, maybe have code
    return np.mean((y_pred - y)**2)




# TODO: train test split
def main():
    n = 100
    x_values, y_values, z_values = helper.generate_data(100)
    x_train, x_test, y_train, y_test, z_train, z_test = helper.train_test_split(
        x_values, y_values, z_values, test_size=0.2)
    

    _, _, beta_OLS = helper.predict_output(
        x_train=x_train, y_train=y_train, z_train=z_train,
        x_test=x_test, y_test=y_test,
        degree=1, regression_method='OLS'
    )


    X_train = helper.create_design_matrix(x_train, y_train, 1)
    X_train_scaled = X_train - np.mean(X_train, axis=0)
    z_train_scaled = z_train - np.mean(z_train, axis=0)
    res, num = SGD(
        theta_init=np.array([[1], [0], [0]]), 
        eta=1e-0, 
        C=cost, 
        n_epochs=10,
        X=X_train_scaled,
        y=z_train_scaled, 
        gamma=0)
    
    
    
    print('---')
    print(f'num: {num}')
    print(beta_OLS)
    print(res)

if __name__ == '__main__':
    main()
