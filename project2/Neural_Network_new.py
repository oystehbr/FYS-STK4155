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

# TODO: ?? delete this
# np.seterr(all='warn')
# np.seterr(over='raise')


# TODO: hyperparameter lmbda

class Neural_Network():
    def __init__(self, no_input_nodes, no_output_nodes, hidden_info):
        """
        Initializing a neural network with the given hyperparameters,
        creating initial weights according to the standard
        normal distribution and set the biases to be 0.01

        :param no_input_nodes (int):
            the number of input variables
        :param no_ouput_nodes (int):
            the number of output variables
        :param hidden info (list(list)):
            # TODO
        """

        # Setting up the Hyperparameters
        self.no_input_nodes = no_input_nodes
        self.no_output_nodes = no_output_nodes
        self.hidden_info = hidden_info

        # Initializing the biases for each layer (exluded the input layer)
        # TODO: set initialize biases instead of the refresh
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
        self.z_list = []

        for [layers, nodes] in self.hidden_info:
            self.z_list.append(np.zeros(
                (layers, no_data, nodes)))

        self.a_list = self.z_list.copy()

        for i, [layers, nodes] in enumerate(self.hidden_info):
            for j in range(layers):
                # If first case
                if i == 0 and j == 0:
                    self.z_list[i][j] = X @ self.input_weights + \
                        self.hidden_bias_list[i][j]
                else:
                    self.z_list[i][j] = self.a_list[i][j-1] @ self.hidden_weights_list[i][j-1] + \
                        self.hidden_bias_list[i][j]

                # Applying the activation function to the hidden layer
                self.a_list[i][j] = self.activation_function_hidden(
                    self.z_list[i][j])

        # Propagate to the output layer
        z_output = self.a_list[-1][-1] @ self.output_weights + self.output_bias

        # Applying the activation function to the output layer (if provided)
        if self.activation_function_output != None:
            y_hat = self.activation_function_output(z_output)
        else:
            y_hat = z_output

        return y_hat

    def backpropagation(self, X, y):
        """
        The backpropagation algorithm for the neural network

        :param X (np.ndarray): 
            the input values
        :param y (np.ndarray): 
            the target values (output values)
        """
        # Batch normalization
        # X -= np.mean(X, axis=0)
        # X /= std(X, axis=0)
        # y -= np.mean(y)

        # Backward propogate through the network
        y_hat = self.feed_forward(X)

        # Error in output
        output_error = y - y_hat

        if self.activation_function_output != None:
            output_delta = self.activation_function_output(
                y_hat, deriv=True) * output_error
        else:
            output_delta = output_error

        output_weights_grad = - self.a_list[-1][-1].T @ output_delta
        output_bias_grad = - np.mean(output_delta, axis=0)

        hidden_bias_grad_list = []
        hidden_weights_grad_list = []

        if len(self.hidden_info) > 1 or self.hidden_info[0][0] > 1:

            last_weights = self.output_weights
            last_delta = output_delta

            # Iterere gjennom layers
            # TODO: set this as self attribute
            reversed_hidden_info = self.hidden_info[::-1]

            for j, [layers, nodes] in enumerate(reversed_hidden_info):
                # Update some values
                hidden_bias_grad = np.zeros((layers, nodes))

                # This happen with the first layer
                hidden_error = last_delta @ last_weights.T
                hidden_delta = self.activation_function_hidden(
                    self.z_list[-(j+1)][-1], deriv=True) * hidden_error

                # If more layers
                if layers > 1:
                    hidden_weights_grad = np.zeros(
                        (layers - 1, nodes, nodes))

                    hidden_delta = self.activation_function_hidden(
                        self.z_list[-(j+1)][-1], deriv=True) * hidden_error
                    hidden_weights_grad_list.append(-
                                                    self.a_list[-(j+1)][-2].T @ last_delta)

                    # Second case, loop through the rest
                    for i in range(layers - 2):
                        hidden_error = hidden_delta @ self.hidden_weights_list[-j][-(
                            i+1)].T

                        hidden_delta = self.activation_function_hidden(
                            self.z_list[j][-(i+2)], deriv=True) * hidden_error

                        hidden_weights_grad[-(i+2)] = - \
                            self.a_list[-(i+3)].T @ hidden_delta

                        hidden_bias_grad[-(j+1)][-(i+1)] = - \
                            np.mean(hidden_delta, axis=0)

                else:
                    hidden_weights_grad = - \
                        self.a_list[-(j+1)][-1].T @ hidden_delta

                hidden_bias_grad[-(j+1)][0] = - \
                    np.mean(hidden_delta, axis=0)

                hidden_weights_grad_list.append(hidden_weights_grad)
                hidden_bias_grad_list.append(hidden_bias_grad)
                last_weights = self.hidden_weights_list[-(j+1)][0]
                last_delta = hidden_delta

            # Siste
            input_error = last_delta @ self.hidden_weights_list[0][0].T

        elif len(self.hidden_info) == 1 and self.hidden_info[0][0] == 1:
            input_error = output_delta @ self.output_weights.T
        else:
            print("Can't handle no hidden layers")

        input_delta = self.activation_function_hidden(
            self.z_list[0][0], deriv=True) * input_error

        hidden_bias_grad_list[0][0] = - np.mean(input_delta, axis=0)
        input_weights_grad = - X.T @ input_delta

        # Regularization
        # TODO: fix this
        # if self.lmbda > 0:
        #     input_weights_grad += self.lmbda*self.input_weights
        #     output_weights_grad += self.lmbda*self.output_weights

        #     # If have more than one hidden layer, then we have hidden_weights too
        #     if self.number_of_hidden_layers > 1:
        #         hidden_weights_grad += self.lmbda*self.hidden_weights

        return input_weights_grad, hidden_weights_grad_list, output_weights_grad, hidden_bias_grad_list, output_bias_grad

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
        v_output_weight = 0
        v_output_bias = 0

        # To be able to have advance hidden structure
        v_hidden_weight_list = [0] * (len(self.hidden_info)*2 + 1)
        v_hidden_bias_list = [0] * (len(self.hidden_info)*2 + 1)

        j = 0
        error_list = []
        accuracy_list = []

        # TODO: change to iteration
        for epoch in range(self.n_epochs):
            for i in range(m):

                # Saving the previous weights and biases
                input_weights_previous = self.input_weights
                output_weights_previous = self.output_weights
                output_bias_previous = self.output_bias

                hidden_bias_previous_list = self.hidden_bias_list
                hidden_weights_previous_list = self.hidden_weights_list

                # Do something with the end interval of the selected
                k = np.random.randint(m)

                # TODO: random integer can be wrongFinding the k-th batch
                xk_batch = X[k*self.M:(k+1)*self.M]
                yk_batch = y[k*self.M:(k+1)*self.M]

                # Finding the different gradients with backpropagation
                input_weights_grad, hidden_weights_grad_list, output_weights_grad, \
                    hidden_bias_grad_list, output_bias_grad = self.backpropagation(
                        xk_batch, yk_batch)

                # Using the gradients and stochastic to update the weights and biases
                v_input_weight = self.gamma*v_input_weight + self.eta*input_weights_grad
                v_output_weight = self.gamma*v_output_weight + self.eta*output_weights_grad
                v_output_bias = self.gamma*v_output_bias + self.eta*output_bias_grad

                print('START')
                print(hidden_weights_grad_list)
                print(hidden_bias_grad_list)
                print('STOP')

                # TODO: comment
                for l in range(len(hidden_weights_grad_list)):
                    v_hidden_weight_list[l] = self.gamma * \
                        v_hidden_weight_list[l] + self.eta * \
                        hidden_weights_grad_list[-(l+1)]

                for u in range(len(hidden_bias_grad_list)):
                    v_hidden_bias_list[u] = self.gamma * \
                        v_hidden_bias_list[u] + self.eta * \
                        hidden_bias_grad_list[-(u+1)]

                # Updating the weights and biases
                self.input_weights = input_weights_previous - v_input_weight
                self.output_weights = output_weights_previous - v_output_weight
                self.output_bias = output_bias_previous - v_output_bias

                for k in range(len(self.hidden_weights_list)):
                    self.hidden_weights_list[k] = hidden_weights_previous_list[k] - \
                        v_hidden_weight_list[k]

                for k1 in range(len(self.hidden_bias_list)):
                    self.hidden_bias_list[k1] = hidden_bias_previous_list[k1] - \
                        v_hidden_bias_list[k1]

                j += 1
                # Checking if the changes are close to 0, then we are done

                # TODO: DODO
                # zero_change = 0
                # zero_change += np.sum(self.input_weights -
                #                       input_weights_previous)

                # zero_change += np.sum(self.hidden_weights -
                #                       hidden_weights_previous)
                # zero_change += np.sum(self.output_weights -
                #                       output_weights_previous)
                # zero_change += np.sum(self.hidden_bias -
                #                       hidden_bias_previous)
                # zero_change += np.sum(self.output_bias -
                #                       output_bias_previous)

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
                # TODO: don't do this, if not specified -> not good to try to fit a model with bad values
                # if sum(abs(y_hat[0] - y_hat) < tol) == len(y_hat) and y_hat[0] != 0:
                #     print(
                #         'PREDICTING THE SAME VALUES, refreshing the weights and biases')
                #     self.refresh_the_biases()
                #     self.refresh_the_weights()
                #     zero_change = 10
                #     # print(v_input_weight)
                #     # print(v_output_weight)
                #     # TODO: is this smart
                #     # v_input_weight = 0
                #     # v_hidden_weight = 0
                #     # v_output_weight = 0
                #     # v_hidden_bias = 0
                #     # v_output_bias = 0
                #     print('-----')

                # TODO: fix
                zero_change = 10
                if zero_change == 0:
                    # TODO: delete printing
                    print('local')
                    print(j)
                    self.error_list = error_list
                    return

        self.error_list = error_list
        # self.accuracy_list = accuracy_list

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

        self.hidden_bias_list = []

        if len(self.hidden_info) >= 1:
            for [layers, nodes] in self.hidden_info:
                hidden_bias = np.zeros(
                    (layers, nodes)) + 0.01

                self.hidden_bias_list.append(hidden_bias)
        else:
            # TODO: raise some error
            print('Need hidden layer to work -> bias')

        self.output_bias = 0.01

    def refresh_the_weights(self):
        """
        Refreshing the weights of the network, the weights will be 
        divided into input, hidden and output weights and every weight are
        drawn from a standard normal distribution.
        """

        self.input_weights = np.random.randn(
            self.no_input_nodes, self.hidden_info[0][1])

        self.hidden_weights_list = []
        # Creating the hidden layer architecture
        if len(self.hidden_info) >= 1:
            for i, [layers, nodes] in enumerate(self.hidden_info):
                # hidden weights inside one structure (same layer in both ends)
                hidden_weights = np.zeros(
                    (layers - 1, nodes, nodes))

                # Amount of layers of the current layers structure
                for index in range(layers - 1):
                    hidden_weights[index] = np.random.randn(
                        nodes,
                        nodes
                    )

                self.hidden_weights_list.append(hidden_weights)

                # Transition to new hidden structure/ output layer
                if i != len(self.hidden_info) - 1:
                    # Send to another hidden layer
                    hidden_weights_transition = np.zeros(
                        (1, nodes, self.hidden_info[i+1][1]))

                self.hidden_weights_list.append(hidden_weights_transition)
        else:
            # TODO: raise some error maybe
            print('NO hidden layers, we need that ')

        # Send it from last hidden layer to outputlayer
        self.output_weights = np.random.randn(
            self.hidden_info[-1][1], self.no_output_nodes)

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


def main(X_train, X_test, y_train, y_test, M=8, n_epochs=3000):
    print("-----STARTING MAIN -----")

    FFNN = Neural_Network(2, 1, 10, 3)
    FFNN.set_activation_function_hidden_layers('sigmoid')
    # FFNN.set_activation_function_output_layer('sigmoid')
    FFNN.set_SGD_values(
        eta=5e-3,
        lmbda=1e-6,
        n_epochs=n_epochs,
        batch_size=M,
        gamma=0.6)
    FFNN.train_model(X_train, y_train)
    FFNN.plot_MSE_of_last_training()
    print("NN done")

    y_hat_train = FFNN.feed_forward(X_train)
    y_hat_test = FFNN.feed_forward(X_test)

    for _y, _y_hat in zip(y_train, y_hat_train):
        diff = abs(_y - _y_hat)
        print(
            f'y_real = {_y[0]: 5.5f},    y_hat = {_y_hat[0]: 5.5f},    diff = {diff[0]: 5.5f}')

    # OLS:
    y_hat_test_OLS, y_hat_train_OLS, _ = helper.predict_output(
        x_train=X_train[:, 0], y_train=X_train[:, 1], z_train=y_train,
        x_test=X_test[:, 0], y_test=X_test[:, 1],
        degree=4, regression_method='OLS'
    )
    print("OLS done")

    # RIDGE:
    y_hat_test_RIDGE, y_hat_train_RIDGE, _ = helper.predict_output(
        x_train=X_train[:, 0], y_train=X_train[:, 1], z_train=y_train,
        x_test=X_test[:, 0], y_test=X_test[:, 1],
        degree=4, regression_method='RIDGE', lmbda=0.01
    )

    print("RIDGE Done")

    print('\n\n\n>>> CHECKING OUR MODEL: \n')
    print('Neural Network:')
    # print('-- (MSE) TRAINING DATA --')
    # print(helper.mean_squared_error(y_train, y_hat_train))
    # print('-- (MSE) TESTING DATA --')
    # print(helper.mean_squared_error(y_test, y_hat_test))
    print(f'-- (R2) TRAINING DATA --')
    print(helper.r2_score(y_hat_train, y_train))
    print(f'-- (R2) TESTING DATA --')
    print(helper.r2_score(y_hat_test, y_test))

    print('\nOLS:')
    print('-- TESTING DATA --')
    # print(helper.mean_squared_error(y_train, y_hat_train_OLS))
    # print('-- TRAINING DATA: --')
    # print(helper.mean_squared_error(y_test, y_hat_test_OLS))
    print(f'-- (R2) TRAINING DATA --')
    print(helper.r2_score(y_hat_train_OLS, y_train))
    print(f'-- (R2) TESTING DATA --')
    print(helper.r2_score(y_hat_test_OLS, y_test))

    print('\nRIDGE:')
    print('-------TESTING DATA-------')
    # print(helper.mean_squared_error(y_train, y_hat_train_RIDGE))
    # print('-------TRAINING DATA:-------')
    # print(helper.mean_squared_error(y_test, y_hat_test_RIDGE))
    print(f'-- (R2) TRAINING DATA --')
    print(helper.r2_score(y_hat_train_RIDGE, y_train))
    print(f'-- (R2) TESTING DATA --')
    print(helper.r2_score(y_hat_test_RIDGE, y_test))


def main2(X_train, X_test, y_train, y_test, M=20, n_epochs=5000):
    FFNN = Neural_Network(2, 1, 2, 1)
    FFNN.set_activation_function_hidden_layers('sigmoid')
    # FFNN.set_activation_function_output_layer('sigmoid')

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


def main3(X_train, X_test, y_train, y_test, M=20, n_epochs=5000):
    """
    Testing without some scaling of the data works!

    """

    print("-----STARTING MAIN -----")

    FFNN = Neural_Network(2, 1, 20, 3)
    FFNN.set_activation_function_hidden_layers('Sigmoid')
    FFNN.set_SGD_values(
        eta=0.01,
        lmbda=0.01,
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


def main4():

    X = np.array([[1, 1], [2, 2], [3, 3]])
    y = np.array([[1], [5], [10]])
    y_scalar = max(y)
    y = y / y_scalar

    FFNN = Neural_Network(2, 1, [[10, 4]])
    FFNN.set_SGD_values(
        n_epochs=5000,
        batch_size=3,
        eta=0.01,
        gamma=0.4,
        lmbda=0)

    FFNN.train_model(X, y)
    FFNN.plot_MSE_of_last_training()

    y_hat = FFNN.feed_forward(X)
    print('PREDICTED')
    print(y_hat)
    print('ACTUAL')
    print(y)


if __name__ == "__main__":
    main4()

    exit()
    # TODO: scaling the data
    # Generate some data from the Franke Function
    x_1, x_2, y = helper.generate_data(100, noise_multiplier=0.1)
    X = np.array(list(zip(x_1, x_2)))
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    main(X_train, X_test, y_train, y_test)

    exit()
    y_scalar = max(y)
    y /= y_scalar
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    # Splitting the data in train and testing
    main2(X_train, X_test, y_train, y_test)
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)
    main2(X_train, X_test, y_train, y_test)
