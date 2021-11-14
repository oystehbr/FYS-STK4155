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

    # If we want the general gradient descent -> two functions in class tho
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


def main2():
    # from tensorflow.keras.layers import Input
    # from tensorflow.keras.models import Sequential
    # from tensorflow.keras.layers import Dense
    # from tensorflow.keras import optimizers
    # from tensorflow.keras import regularizers
    # from tensorflow.keras.utils import to_categorical

    return
    X = np.array([[1, 1], [2, 2], [3, 3]])
    X = X/np.max(X)
    y = np.array(([2], [4], [6]))
    y = y/10
    epochs = 5000
    batch_size = 3
    n_neurons_layer = 20
    n_categories = 1

    FFNN = Neural_Network(2, 1, n_neurons_layer, 1)
    FFNN.train_model(
        X, y,
        eta=0.05, n_epochs=epochs,
        M=batch_size, gamma=0.1)

    def create_neural_network_keras(n_neurons_layer1, n_neurons_layer2, n_categories, eta, lmbd):
        model = Sequential()
        model.add(Dense(n_neurons_layer1, activation='sigmoid',
                  kernel_regularizer=regularizers.l2(lmbd)))
        model.add(Dense(n_neurons_layer2, activation='sigmoid',
                  kernel_regularizer=regularizers.l2(lmbd)))
        model.add(Dense(n_categories, activation='sigmoid'))
        sgd = optimizers.SGD(learning_rate=eta)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd, metrics=['accuracy'])
        return model

    DNN = create_neural_network_keras(n_neurons_layer, n_neurons_layer, n_categories,
                                      eta=0.05, lmbd=0.1)

    DNN.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    scores = DNN.evaluate(X, y)

    y_hat = FFNN.feed_forward(X)

    print(y)
    print(scores)
    print(y_hat)
