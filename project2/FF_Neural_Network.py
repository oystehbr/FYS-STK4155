"""
Created a class to be able to create a Neural Network
where the structure of the network will be determined by
the initial values we feed into the constructor.

The weights in the network will be initalizied by the normal distribution
and the biases will be initialized to 0.01
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import helper
from activation_functions import sigmoid, RELU, Leaky_RELU, soft_max, sigmoid_classification
from sklearn.metrics import accuracy_score
# TODO: ?? delete this
# np.seterr(all='warn')
# np.seterr(over='raise')

# TODO: does the CLASS work for more than 1 output nodes

# TODO: hyperparameter lmbda


class Neural_Network():
    def __init__(self, no_input_nodes, no_output_nodes, no_hidden_nodes, no_hidden_layers):
        """
        Initializing a neural network with the given hyperparameters,
        creating initial weights according to the standard
        normal distribution and set the biases to be 0.01

        :param no_input_nodes (int):
            the number of input variables
        :param no_ouput_nodes (int):
            the number of output variables
        :param no_hidden_nodes (int):
            the amount of nodes inside each hidden layer
        :param no_hidden_layers (int):
            the number of hidden layers
        """

        # Setting up the Hyperparameters
        self.no_input_nodes = no_input_nodes
        self.no_ouput_nodes = no_output_nodes
        self.no_hidden_nodes = no_hidden_nodes
        self.number_of_hidden_layers = no_hidden_layers

        # Initializing the biases for each layer (exluded the input layer)
        self.refresh_the_biases()

        # Initialize weights using the standard normal distribution
        self.refresh_the_weights()

        # Initializing some SGD_values (can be reinitialized)
        self.set_SGD_values()

        # Initializing the activation function used (can be reinitialized)
        self.set_activation_function_hidden_layers()
        self.set_activation_function_output_layer()

    def feed_forward(self, X):
        """
        Running through our network with the input values X

        :param X (np.ndarray):
            input variables (we want to propagate through the network with)

        :return (np.ndarray):
            the predicted output value(s) given our current weights and biases
        """

        # Find the number of data for creating the z and a - arrays of right dim
        no_data = X.shape[0]

        # Initializing arrays for the values of each layer
        self.z = np.zeros(
            (self.number_of_hidden_layers, no_data, self.no_hidden_nodes))
        self.a = self.z.copy()  # copy (same dimension on the activation-array)

        # TODO: ?? maybe working (above)
        # self.a = np.zeros((self.number_of_hidden_layers,
        #                   no_data, self.no_hidden_nodes))

        # Iterate through the hidden layers
        for i in range(self.number_of_hidden_layers):
            if i == 0:
                self.z[i] = X @ self.input_weights + self.hidden_bias[i]
            else:
                self.z[i] = self.a[i-1] @ self.hidden_weights[i-1] + \
                    self.hidden_bias[i]

            # Applying the activation function to the hidden layer
            self.a[i] = self.activation_function_hidden(self.z[i])

        # Propagate to the output layer
        self.z_output = self.a[-1] @ self.output_weights + self.output_bias

        # Applying the activation function to the output layer (if provided)
        if self.activation_function_output != None:
            y_hat = self.activation_function_output(self.z_output)
        else:
            y_hat = self.z_output

        return y_hat

    def backpropagation(self, X, y):
        """
        The backpropagation algorithm for the neural network

        :param X (np.ndarray): 
            the input values
        :param y (np.ndarray): 
            the target values (output values)
        """

        # Backward propogate through the network
        y_hat = self.feed_forward(X)

        # Error in output
        output_error = y - y_hat

        if self.activation_function_output != None:
            output_delta = self.activation_function_output(
                y_hat, deriv=True) * output_error
        else:
            output_delta = output_error

        output_weights_grad = - self.a[-1].T @ output_delta
        output_bias_grad = - np.mean(output_delta, axis=0)
        hidden_bias_grad = np.zeros(
            (self.number_of_hidden_layers, self.no_hidden_nodes))

        if self.number_of_hidden_layers > 1:
            hidden_weights_grad = np.zeros(
                (self.number_of_hidden_layers - 1, self.no_hidden_nodes, self.no_hidden_nodes))
            hidden_error = output_delta @ self.output_weights.T
            hidden_delta = self.activation_function_hidden(
                self.z[-1], deriv=True) * hidden_error
            hidden_weights_grad[-1] = - self.a[-2].T @ output_delta

            for i in range(self.number_of_hidden_layers - 2):
                # endret
                hidden_error = hidden_delta @ self.hidden_weights[-(i+1)].T
                hidden_delta = self.activation_function_hidden(
                    self.z[-(i+2)], deriv=True) * hidden_error
                hidden_weights_grad[-(i+2)] = - self.a[-(i+3)].T @ hidden_delta
                hidden_bias_grad[-(i+2)] = - np.mean(hidden_delta, axis=0)

            input_error = hidden_delta @ self.hidden_weights[0].T

        elif self.number_of_hidden_layers == 1:
            input_error = output_delta @ self.output_weights.T
            hidden_weights_grad = 0

        input_delta = self.activation_function_hidden(
            self.z[0], deriv=True) * input_error

        hidden_bias_grad[0] = - np.mean(input_delta, axis=0)
        input_weights_grad = - X.T @ input_delta

        # Regularization
        if self.lmbda > 0:
            input_weights_grad += self.lmbda*self.input_weights
            output_weights_grad += self.lmbda*self.output_weights

            # If have more than one hidden layer, then we have hidden_weights too
            if self.number_of_hidden_layers > 1:
                hidden_weights_grad += self.lmbda*self.hidden_weights

        return input_weights_grad, hidden_weights_grad, output_weights_grad, hidden_bias_grad, output_bias_grad

    def SGD(self, X, y, tol=1e-3):
        """
        The stochastic gradient descent algorithm for updating 
        the weights and the biases of the network. It will use the 
        initialized SGD_values. 

        :param X (np.ndarray): 
            the input values
        :param y (np.ndarray): 
            the target values (output values)
        :param tol (number): 
            # TODO: delete or implement this
            some tolerance to check if the changes in weights and biases are so small,
            so it's not necessary to adjust the weights, biases anymore.
        """

        # TODO: change
        m = int(X.shape[0]/self.M)

        # Starting with zero momentum in the gradient descent
        v_input_weight = 0
        v_hidden_weight = 0
        v_output_weight = 0
        v_hidden_bias = 0
        v_output_bias = 0

        j = 0
        error_list = []
        accuracy_list = []

        # TODO: change to iteration
        for epoch in range(self.n_epochs):
            for i in range(m):

                # Saving the previous weights and biases
                input_weights_previous = self.input_weights
                hidden_weights_previous = self.hidden_weights
                output_weights_previous = self.output_weights
                hidden_bias_previous = self.hidden_bias
                output_bias_previous = self.output_bias

                # Do something with the end interval of the selected
                k = np.random.randint(m)

                # TODO: random integer can be wrongFinding the k-th batch
                xk_batch = X[k*self.M:(k+1)*self.M]
                yk_batch = y[k*self.M:(k+1)*self.M]

                # Finding the different gradients with backpropagation
                input_weights_grad, hidden_weights_grad, output_weights_grad, \
                    hidden_bias_grad, output_bias_grad = self.backpropagation(
                        xk_batch, yk_batch)

                # Using the gradients and stochastic to update the weights and biases
                v_input_weight = self.gamma*v_input_weight + self.eta*input_weights_grad
                v_hidden_weight = self.gamma*v_hidden_weight + self.eta*hidden_weights_grad
                v_output_weight = self.gamma*v_output_weight + self.eta*output_weights_grad
                v_hidden_bias = self.gamma*v_hidden_bias + self.eta*hidden_bias_grad
                v_output_bias = self.gamma*v_output_bias + self.eta*output_bias_grad

                # Updating the weights and biases
                self.input_weights = input_weights_previous - v_input_weight
                self.hidden_weights = hidden_weights_previous - v_hidden_weight
                self.output_weights = output_weights_previous - v_output_weight
                self.hidden_bias = hidden_bias_previous - v_hidden_bias
                self.output_bias = output_bias_previous - v_output_bias
                j += 1
                # Checking if the changes are close to 0, then we are done
                zero_change = 0
                zero_change += np.sum(self.input_weights -
                                      input_weights_previous)

                zero_change += np.sum(self.hidden_weights -
                                      hidden_weights_previous)
                zero_change += np.sum(self.output_weights -
                                      output_weights_previous)
                zero_change += np.sum(self.hidden_bias -
                                      hidden_bias_previous)
                zero_change += np.sum(self.output_bias -
                                      output_bias_previous)

                # Check if we have reached the tolerance
                # if np.sum(np.abs(theta_next - theta_previous)) < tol:
                #     print('local')
                #     return theta_next, j

                # Keeping track of the mean square error after each iteration.
                # TODO: make this a method
                y_hat = self.feed_forward(X)
                error_list.append(helper.mean_squared_error(
                    y_hat, y))

                # TODO: make some input if this shall happen -> takes more time
                # the_activation_function_output = self.activation_function_output
                # self.set_activation_function_output_layer(
                #     'sigmoid_classification')
                # accuracy_list.append(accuracy_score(self.feed_forward(X), y))
                # # Reset the activation function output layer
                # self.activation_function_output = the_activation_function_output

                # Checking if the predicted values are the same, if so - restart the weights and biases
                if sum(abs(y_hat[0] - y_hat) < tol) == len(y_hat) and y_hat[0] != 0:
                    print(
                        'PREDICTING THE SAME VALUES, refreshing the weights and biases')
                    self.refresh_the_biases()
                    self.refresh_the_weights()
                    zero_change = 10

                if zero_change == 0:
                    # TODO: delete printing
                    print('local')
                    print(j)
                    self.error_list = error_list
                    return

        self.error_list = error_list
        self.accuracy_list = accuracy_list

    def plot_MSE_of_last_training(self):
        """
        Plot the mean square error vs. iterations of the last
        training
        """

        plt.loglog(range(len(self.error_list)), self.error_list)
        plt.show()

    def plot_accuracy_score_last_training(self):
        """
        Plot the mean square error vs. iterations of the last
        training
        """
        # TODO: maybe add some title or something?
        plt.title(
            'Accuracy score from the last training. \nAccuracy = 1 (100% correct)')
        plt.ylabel('The accuracy between 0 - 1')
        plt.xlabel('The number of iterations')
        plt.plot(range(len(self.accuracy_list)), self.accuracy_list)
        plt.show()

    def refresh_the_biases(self):
        """
        Refreshing the biases of the network, the biases will be 
        divided into hidden and output biases and every bias are
        set to be 0.01
        """

        self.hidden_bias = np.zeros(
            (self.number_of_hidden_layers, self.no_hidden_nodes)) + 0.01

        self.output_bias = 0.01

    def refresh_the_weights(self):
        """
        Refreshing the weights of the network, the weights will be 
        divided into input, hidden and output weights and every weight are
        drawn from a standard normal distribution.
        """

        self.input_weights = np.random.randn(
            self.no_input_nodes, self.no_hidden_nodes)

        if self.number_of_hidden_layers > 1:
            self.hidden_weights = np.zeros(
                (self.number_of_hidden_layers - 1, self.no_hidden_nodes, self.no_hidden_nodes))
            for i in range(self.number_of_hidden_layers - 1):
                self.hidden_weights[i] = np.random.randn(
                    self.no_hidden_nodes,
                    self.no_hidden_nodes
                )
        else:
            self.hidden_weights = 0

        self.output_weights = np.random.randn(
            self.no_hidden_nodes, self.no_ouput_nodes)

    def set_SGD_values(self, eta: float = 0.05, lmbda: float = 0, n_epochs: int = 1000, batch_size: int = 3, gamma: float = 0.5):
        """
        Method for setting the values that the stochastic gradient descent
        will be using in its algorithm.

        :param eta (number):
            the learning rate
        :param lmbda (number):
            the regularization parameter
        :param n_epochs (int):
            # TODO:
        :param batch_size (int):
            the data are split into batches (for the stochastic). If
            the size of the batch is equal to the size of the input data,
            then we will have no stochastic.
        :param gamma (number):
            the amount of momentum we will be using in the gradient descent
            descent. If gamma is 0, then we will have no momentum. If gamma is 1, then
            we will use "full" momentum instead.

        """

        self.eta = eta
        self.lmbda = lmbda
        self.n_epochs = n_epochs
        # TODO: maybe change to batch_size
        self.M = batch_size
        self.gamma = gamma

    def set_activation_function_hidden_layers(self, activation_name: str = 'sigmoid'):
        """
        Setting the activation function for the hidden layers.

        :param activation_name (str), default = 'sigmoid':
            the preffered activation function: sigmoid, Leaky_RELU, RELU or softmax
        """

        if activation_name.lower() == 'sigmoid'.lower():
            self.activation_function_hidden = sigmoid
        elif activation_name.lower() == 'Leaky_RELU'.lower():
            self.activation_function_hidden = Leaky_RELU
        elif activation_name.lower() == 'RELU'.lower():
            self.activation_function_hidden = RELU
        elif activation_name.lower() == 'softmax'.lower():
            self.activation_function_hidden = soft_max
        else:
            print('Not a proper activation function')

    def set_activation_function_output_layer(self, activation_name: str = ''):
        """
        Setting the activation function for the output layer.

        :param activation_name (str), default = 'sigmoid':
            the preffered activation function: sigmoid, Leaky_RELU, RELU, softmax, sigmoid_classification
        """

        if activation_name.lower() == 'sigmoid'.lower():
            self.activation_function_output = sigmoid
        elif activation_name.lower() == 'Leaky_RELU'.lower():
            self.activation_function_output = Leaky_RELU
        elif activation_name.lower() == 'RELU'.lower():
            self.activation_function_output = RELU
        elif activation_name.lower() == 'softmax'.lower():
            self.activation_function_output = soft_max
        elif activation_name.lower() == 'sigmoid_classification'.lower():
            self.activation_function_output = sigmoid_classification
        elif activation_name.lower() == '':
            self.activation_function_output = None
        else:
            print('Not a proper activation function')

    def train_model(self, X, y):
        """
        This function will train the model by running through
        a stochastic gradient descent algorithm (with help from backpropagation)
        to update the weights and biases in the neural network

        """

        self.SGD(X, y)


def main(X_train, X_test, y_train, y_test, M, n_epochs):
    print("-----STARTING MAIN -----")

    FFNN = Neural_Network(2, 1, 2, 1)
    FFNN.set_activation_function_hidden_layers('Leaky_RELU')
    FFNN.set_activation_function_output_layer('Leaky_RELU')
    FFNN.set_SGD_values(
        n_epochs=n_epochs,
        batch_size=M)
    FFNN.train_model(X_train, y_train)

    y_hat_train = FFNN.feed_forward(X_train)
    y_hat_test = FFNN.feed_forward(X_test)

    for _y, _y_hat in zip(y_train, y_hat_train):
        diff = abs(_y - _y_hat)
        print(
            f'y_real = {_y[0]: 5.5f},    y_hat = {_y_hat[0]: 5.5f},    diff = {diff[0]: 5.5f}')

    # # OLS:
    # y_hat_test_OLS, y_hat_train_OLS, _ = helper.predict_output(
    #     x_train=X_train[:, 0], y_train=X_train[:, 1], z_train=y_train,
    #     x_test=X_test[:, 0], y_test=X_test[:, 1],
    #     degree=4, regression_method='OLS'
    # )

    # # RIDGE:
    # y_hat_test_RIDGE, y_hat_train_RIDGE, _ = helper.predict_output(
    #     x_train=X_train[:, 0], y_train=X_train[:, 1], z_train=y_train,
    #     x_test=X_test[:, 0], y_test=X_test[:, 1],
    #     degree=4, regression_method='RIDGE', lmbda=0.01
    # )

    print('\n\n\n>>> CHECKING OUR MODEL: \n')
    print('Neural Network:')
    print('-- (MSE) TRAINING DATA --')
    print(helper.mean_squared_error(y_train, y_hat_train))
    print('-- (MSE) TESTING DATA --')
    print(helper.mean_squared_error(y_test, y_hat_test))
    print(f'-- (R2) TESTING DATA --')
    print(helper.r2_score(y_hat_test, y_test))

    # print('\nOLS:')
    # print('-- TESTING DATA --')
    # print(helper.mean_squared_error(y_train, y_hat_train_OLS))
    # print('-- TRAINING DATA: --')
    # print(helper.mean_squared_error(y_test, y_hat_test_OLS))
    # print(f'-- (R2) TESTING DATA --')
    # print(helper.r2_score(y_hat_test_OLS, y_test))

    # print('\nRIDGE:')
    # print('-------TESTING DATA-------')
    # print(helper.mean_squared_error(y_train, y_hat_train_RIDGE))
    # print('-------TRAINING DATA:-------')
    # print(helper.mean_squared_error(y_test, y_hat_test_RIDGE))
    # print(f'-- (R2) TESTING DATA --')
    # print(helper.r2_score(y_hat_test_RIDGE, y_test))


def main2(X_train, X_test, y_train, y_test):
    FFNN = Neural_Network(2, 1, 5, 1)
    FFNN.set_activation_function_hidden_layers('RELU')
    FFNN.set_activation_function_output_layer('RELU')

    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([[2], [4], [6]])
    y_scalar = max(y)
    y = y / y_scalar

    FFNN.set_SGD_values(
        n_epochs=2000,
        batch_size=1,
        gamma=0.8)

    FFNN.train_model(X, y)
    y_hat = FFNN.feed_forward(X)
    for _y, _y_hat in zip(y, y_hat):
        diff = abs(_y - _y_hat)
        print(
            f'y_real = {_y[0]: 5.5f},    y_hat = {_y_hat[0]: 5.5f},    diff = {diff[0]: 5.5f}')

    print(y)
    print(y_hat)


def main3(X_train, X_test, y_train, y_test, M=3, n_epochs=5000):
    """
    Testing without some scaling of the data works!

    """

    print("-----STARTING MAIN -----")

    FFNN = Neural_Network(2, 1, 10, 2)
    FFNN.set_activation_function_hidden_layers('Sigmoid')
    FFNN.set_SGD_values(
        eta=0.01,
        lmbda=0.001,
        n_epochs=n_epochs,
        batch_size=M,
        gamma=0.7)
    FFNN.train_model(X_train, y_train)
    FFNN.plot_MSE_of_last_training()

    y_hat_train = FFNN.feed_forward(X_train)
    y_hat_test = FFNN.feed_forward(X_test)

    for _y, _y_hat in zip(y_train, y_hat_train):
        diff = abs(_y - _y_hat)
        print(
            f'y_real = {_y[0]: 5.5f},    y_hat = {_y_hat[0]: 5.5f},    diff = {diff[0]: 5.5f}')

    print('\n\n\n>>> CHECKING OUR MODEL: \n')
    print('Neural Network:')
    print('>> (MSE) TRAINING DATA: ', end='')
    print(helper.mean_squared_error(y_train, y_hat_train))
    print('>> (MSE) TESTING DATA: ', end='')
    print(helper.mean_squared_error(y_test, y_hat_test))
    print('>> (R2) TRAINING DATA: ', end='')
    print(helper.r2_score(y_train, y_hat_train))
    print(f'>> (R2) TESTING DATA: ', end='')
    print(helper.r2_score(y_hat_test, y_test))


if __name__ == "__main__":

    # TODO: scaling the data
    # Generate some data from the Franke Function
    x_1, x_2, y = helper.generate_data(50, noise_multiplier=0.2)
    X = np.array(list(zip(x_1, x_2)))
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    main3(X_train, X_test, y_train, y_test, )

    exit()
    y_scalar = max(y)
    y /= y_scalar
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    # Splitting the data in train and testing
<< << << < HEAD
main2(X_train, X_test, y_train, y_test)
== == == =
X_train, X_test, y_train, y_test = helper.train_test_split(X, y)
main(X_train, X_test, y_train, y_test)
>>>>>> > d92d42c638e7a4645979ed6741149d65754c3e26
