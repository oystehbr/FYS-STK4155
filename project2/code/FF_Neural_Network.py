"""
Created a class to be able to create a Neural Network
where the structure of the network will be determined by
the initial values we feed into the constructor.

The weights in the network will be initalizied by the normal distribution
and the biases will be initialized to 0.01
"""

from activation_functions import sigmoid, RELU, Leaky_RELU, sigmoid_classification
from autograd import elementwise_grad as egrad
from cost_functions import MSE
import autograd.numpy as np
import matplotlib.pyplot as plt
import helper


class Neural_Network():
    def __init__(self, no_input_nodes, no_output_nodes, node_list):
        """
        Initializing a neural network with the given hyperparameters,
        creating initial weights according to the standard
        normal distribution and set the biases to be 0.01

        :param no_input_nodes (int):
            the number of input variables
        :param no_ouput_nodes (int):
            the number of output variables
        :param node_list (list):
            node_list[i], explains the amount of nodes of the (i+1)-th hidden layer

            e.g. node_list = [2, 3]. 2 hidden layers, with 2 and 3 nodes (respectively)
        """

        # Setting up the Hyperparameters
        self.no_input_nodes = no_input_nodes
        self.no_output_nodes = no_output_nodes
        self.node_list = node_list

        # Initializing the weights and biases
        self.initialize_the_weights()
        self.initialize_the_biases()

        # Initializing some SGD_values (can be reinitialized)
        self.set_SGD_values(eta=0.05, lmbda=0, n_epochs=1000,
                            batch_size=3, gamma=0.5)

        # Initializing the activation function used (can be reinitialized)
        self.set_activation_function_hidden_layers()
        self.set_activation_function_output_layer()

        # Default cost-function is MSE
        self.set_cost_function(MSE)

    def feed_forward(self, X):
        """
        Running through our network with the input values X

        :param X (np.ndarray):
            input variables (we want to propagate through the network with)

        :return (np.ndarray):
            the predicted output value(s) given our current weights and biases
        """

        # Initializing arrays for the values of each layer
        self.z = [0] * len(self.node_list)

        self.a = self.z.copy()  # copy (same dimension on the activation-array)

        # Iterate through the hidden layers
        for i in range(len(self.node_list)):
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
            self.y_hat = self.activation_function_output(self.z_output)
        else:
            self.y_hat = self.z_output
        return self.y_hat

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

        output_error = self.cost_grad(y_hat, y)

        if self.activation_function_output != None:
            output_delta = self.activation_function_output(
                y_hat, deriv=True) * output_error
        else:
            output_delta = output_error

        output_weights_grad = self.a[-1].T @ output_delta
        output_bias_grad = np.mean(output_delta, axis=0)
        # hidden_bias_grad = np.zeros(
        #     (self.number_of_hidden_layers, self.no_hidden_nodes))
        hidden_bias_grad = []
        for nodes in self.node_list:
            hidden_bias_grad.append(np.zeros(nodes))
        number_of_hidden_layers = len(self.node_list)

        if number_of_hidden_layers > 1:
            # hidden_weights_grad = np.zeros(
            #     (self.number_of_hidden_layers - 1, self.no_hidden_nodes, self.no_hidden_nodes))
            hidden_weights_grad = [0] * (number_of_hidden_layers - 1)
            hidden_error = output_delta @ self.output_weights.T
            hidden_delta = self.activation_function_hidden(
                self.z[-1], deriv=True) * hidden_error
            hidden_weights_grad[-1] = self.a[-2].T @ hidden_delta

            for i in range(number_of_hidden_layers - 2):
                hidden_error = hidden_delta @ self.hidden_weights[-(i+1)].T
                hidden_delta = self.activation_function_hidden(
                    self.z[-(i+2)], deriv=True) * hidden_error
                hidden_weights_grad[-(i+2)] = self.a[-(i+3)].T @ hidden_delta
                hidden_bias_grad[-(i+2)] = np.mean(hidden_delta, axis=0)

            input_error = hidden_delta @ self.hidden_weights[0].T

        elif number_of_hidden_layers == 1:
            input_error = output_delta @ self.output_weights.T
            hidden_weights_grad = [0]

        input_delta = self.activation_function_hidden(
            self.z[0], deriv=True) * input_error

        hidden_bias_grad[0] = np.mean(input_delta, axis=0)
        input_weights_grad = X.T @ input_delta

        # Regularization
        if self.lmbda > 0:
            input_weights_grad += self.lmbda*self.input_weights
            output_weights_grad += self.lmbda*self.output_weights

            # If have more than one hidden layer, then we have hidden_weights too
            if number_of_hidden_layers > 1:
                #hidden_weights_grad += self.lmbda*self.hidden_weights
                hidden_weights_grad = [hidden_weights_grad[k] + self.lmbda *
                                       weight for k, weight in enumerate(self.hidden_weights)]

        return input_weights_grad, hidden_weights_grad, output_weights_grad, hidden_bias_grad, output_bias_grad

    def SGD(self, X, y, tol=1e-4):
        """
        The stochastic gradient descent algorithm for updating 
        the weights and the biases of the network. It will use the 
        initialized SGD_values. 

        :param X (np.ndarray): 
            the input values
        :param y (np.ndarray): 
            the target values (output values)
        :param tol (number): 
            if the model are predicting the same values by a tolerance, 
            then a message will occure 
        """

        # Finding the number of minibatches
        num_of_minibatches = int(X.shape[0]/self.batch_size)

        # Starting with zero momentum in the gradient descent
        v_input_weight = 0
        v_hidden_weight = [0] * (len(self.node_list)-1)
        v_output_weight = 0
        v_hidden_bias = [0] * (len(self.node_list))
        v_output_bias = 0

        iter = 0
        error_list = []
        accuracy_list = []

        for epoch in range(self.n_epochs):
            for i in range(num_of_minibatches):

                # Getting some random batch no. k
                k = np.random.randint(num_of_minibatches)

                xk_batch = X[k*self.batch_size:(k+1)*self.batch_size]
                yk_batch = y[k*self.batch_size:(k+1)*self.batch_size]

                # Finding the different gradients with backpropagation
                input_weights_grad, hidden_weights_grad, output_weights_grad, \
                    hidden_bias_grad, output_bias_grad = self.backpropagation(
                        xk_batch, yk_batch)

                # Using the gradients and stochastic to update the weights and biases
                v_input_weight = self.gamma*v_input_weight + self.eta*input_weights_grad
                # v_hidden_weight = self.gamma*v_hidden_weight + self.eta*hidden_weights_grad

                v_hidden_weight = [self.gamma*weight + self.eta*grad for weight,
                                   grad in zip(v_hidden_weight, hidden_weights_grad)]
                v_output_weight = self.gamma*v_output_weight + self.eta*output_weights_grad
                v_hidden_bias = [self.gamma*bias + self.eta*grad for bias,
                                 grad in zip(v_hidden_bias, hidden_bias_grad)]
                v_output_bias = self.gamma*v_output_bias + self.eta*output_bias_grad

                # Updating the weights and biases
                self.input_weights -= v_input_weight
                # self.hidden_weights -= v_hidden_weight
                self.hidden_weights = [
                    weight - v for weight, v in zip(self.hidden_weights, v_hidden_weight)]
                self.output_weights -= v_output_weight
                # self.hidden_bias -= v_hidden_bias
                self.hidden_bias = [bias - v for bias,
                                    v in zip(self.hidden_bias, v_hidden_bias)]
                self.output_bias -= v_output_bias

                iter += 1

                # If we want to save the error's according to the costfunction
                if self.keep_cost_values:
                    error_list.append(self.cost_function(
                        self.feed_forward(X), y))

                # If we want to save the accuracy score of our model
                if self.keep_accuracy_score:
                    the_activation_function_output = self.activation_function_output
                    self.set_activation_function_output_layer(
                        'sigmoid_classification')
                    accuracy_list.append(
                        helper.accuracy_score(self.feed_forward(X), y))
                    # Reset the activation function output layer
                    self.activation_function_output = the_activation_function_output

                # # Is able to save us from local minimum, but commented out (not used anymore)
                # if sum(abs(self.y_hat[0] - self.y_hat) < tol) == len(self.y_hat) and self.y_hat[0] != 0 and not self.keep_accuracy_score:
                #     print('>> Predicting all same values, unstable values')
                #     self.initialize_the_biases()
                #     self.initialize_the_weights()
                # #     exit()

        self.error_list = error_list
        self.accuracy_list = accuracy_list

    def plot_cost_of_last_training(self):
        """
        Plot the cost value vs. iterations of the last
        training
        """

        plt.title(
            'Value of the cost-function from the last training.')
        plt.ylabel('Cost value')
        plt.xlabel('The number of iterations')
        plt.loglog(range(len(self.error_list)), self.error_list)
        plt.savefig(f'plots/cost_of_last_training_{self.gamma}_{self.eta}.png')
        plt.show()

    def plot_accuracy_score_last_training(self):
        """
        Plot the mean square error vs. iterations of the last
        training
        """

        plt.title(
            'Accuracy score from the last training. \nAccuracy = 1 (100% correct)')
        plt.ylabel('The accuracy between 0 - 1')
        plt.xlabel('The number of iterations')
        plt.plot(range(len(self.accuracy_list)), self.accuracy_list)
        plt.savefig(
            f'plots/accuracy_of_last_training_{self.gamma}_{self.eta}.png')
        plt.show()

    def initialize_the_biases(self):
        """
        Initializing the biases of the network, the biases will be 
        divided into hidden and output biases and every bias are
        set to be 0.01
        """
        self.hidden_bias = []
        for i in range(len(self.node_list)):
            self.hidden_bias.append(np.zeros(self.node_list[i]) + 0.01)

        # self.hidden_bias = np.zeros(
        #     (self.number_of_hidden_layers, self.no_hidden_nodes)) + 0.01

        self.output_bias = 0.01

    def initialize_the_weights(self):
        """
        Initializing the weights of the network, the weights will be 
        divided into input, hidden and output weights and every weight are
        drawn from a standard normal distribution.
        """

        self.input_weights = np.random.randn(
            self.no_input_nodes, self.node_list[0])

        self.hidden_weights = []
        for i in range(len(self.node_list) - 1):
            self.hidden_weights.append(np.random.randn(
                self.node_list[i], self.node_list[i+1]))

        self.output_weights = np.random.randn(
            self.node_list[-1], self.no_output_nodes)

    def set_SGD_values(self, eta: float = None, lmbda: float = None, n_epochs: int = None, batch_size: int = None, gamma: float = None):
        """
        Method for setting the values that the stochastic gradient descent
        will be using in its algorithm.

        :param eta (number):
            the learning rate
        :param lmbda (number):
            the regularization parameter
        :param n_epochs (int):
            number of epochs to iterate through
        :param batch_size (int):
            the data are split into batches (for the stochastic). If
            the size of the batch is equal to the size of the input data,
            then we will have no stochastic.
        :param gamma (number):
            the amount of momentum we will be using in the gradient descent
            descent. If gamma is 0, then we will have no momentum. If gamma is 1, then
            we will use "full" momentum instead.

        """
        if eta != None:
            self.eta = eta

        if lmbda != None:
            self.lmbda = lmbda

        if n_epochs != None:
            self.n_epochs = n_epochs

        if batch_size != None:
            self.batch_size = batch_size

        if gamma != None:
            self.gamma = gamma

    def set_activation_function_hidden_layers(self, activation_name: str = 'sigmoid'):
        """
        Setting the activation function for the hidden layers.

        :param activation_name (str), default = 'sigmoid':
            the preffered activation function: sigmoid, Leaky_RELU or RELU
        """

        if activation_name.lower() == 'sigmoid'.lower():
            self.activation_function_hidden = sigmoid
        elif activation_name.lower() == 'Leaky_RELU'.lower():
            self.activation_function_hidden = Leaky_RELU
        elif activation_name.lower() == 'RELU'.lower():
            self.activation_function_hidden = RELU
        else:
            print('Not a proper activation function')

    def set_activation_function_output_layer(self, activation_name: str = ''):
        """
        Setting the activation function for the output layer.

        :param activation_name (str), default = '':
            the preffered activation function: sigmoid, Leaky_RELU, RELU or sigmoid_classification
        """

        if activation_name.lower() == 'sigmoid'.lower():
            self.activation_function_output = sigmoid
        elif activation_name.lower() == 'Leaky_RELU'.lower():
            self.activation_function_output = Leaky_RELU
        elif activation_name.lower() == 'RELU'.lower():
            self.activation_function_output = RELU
        elif activation_name.lower() == 'sigmoid_classification'.lower():
            self.activation_function_output = sigmoid_classification
        elif activation_name.lower() == '':
            self.activation_function_output = None
        else:
            print('Not a proper activation function')

    def set_cost_function(self, cost_function):
        """
        Setting the costfunction we want the Neural Network to 
        optimize against and creating an autograd object

        :param cost_function (function): 
            the costfunction to use in the backpropagation
        """

        self.cost_function = cost_function
        self.cost_grad = egrad(cost_function)

    def train_model(self, X, y, keep_cost_values: bool = False, keep_accuracy_score: bool = False):
        """
        This function will train the model by running through
        a stochastic gradient descent algorithm (with help from backpropagation)
        to update the weights and biases in the neural network

        :param X (np.ndarray):
            the input variable we want to train the network with
        :param y (np.ndarray):
            the target values we want to train the model to predict
        :param keep_cost_values (bool):
            - False (default): do not save anything
            - True: saves what the cost value is after running each backpropagation,
                so it is possible to plot the cost value vs. training time

        :param keep_accuracy_score (bool):
            - False (default): do not save anything
            - True: saves what the accuracy score is after running each backpropagation,
                so it is possible to plot the accuracy vs. training time
        """

        self.keep_cost_values = keep_cost_values
        self.keep_accuracy_score = keep_accuracy_score

        # Calling the SGD-function to help us train the model
        self.SGD(X, y)


def main(X_train, X_test, y_train, y_test, M=15, n_epochs=1000):
    print("-----STARTING MAIN -----")

    node_list = [30]*4
    FFNN = Neural_Network(2, 1, node_list)
    FFNN.set_activation_function_hidden_layers('sigmoid')
    FFNN.set_SGD_values(
        eta=5e-4,
        lmbda=1e-6,
        n_epochs=n_epochs,
        batch_size=M,
        gamma=0.8)
    FFNN.set_cost_function(MSE)
    FFNN.train_model(X_train, y_train)
    FFNN.plot_cost_of_last_training()

    y_hat_train = FFNN.feed_forward(X_train)
    y_hat_test = FFNN.feed_forward(X_test)

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

    for _ in y_hat_train:
        print(_)


def main3(X_train, X_test, y_train, y_test, M=10, n_epochs=1000):
    """
    Testing without some scaling of the data works!

    """

    print("-----STARTING MAIN -----")

    node_list = [20]*3
    FFNN = Neural_Network(2, 1, node_list)
    FFNN.set_activation_function_hidden_layers('Leaky_RELU')
    FFNN.set_SGD_values(
        eta=1e-4,
        lmbda=1e-1,
        n_epochs=n_epochs,
        batch_size=M,
        gamma=0.5)
    FFNN.train_model(X_train, y_train, keep_cost_values=True)
    FFNN.plot_cost_of_last_training()

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
    y = np.array([[2], [4], [6]])
    y_scalar = max(y)
    y = y / y_scalar

    node_list = [2, 2, 2]
    FFNN = Neural_Network(2, 1, node_list)
    FFNN.set_SGD_values(
        n_epochs=4000,
        batch_size=3,
        eta=0.1,
        gamma=0.8,
        lmbda=0)
    FFNN.set_cost_function(MSE)
    FFNN.train_model(X, y, keep_cost_values=True)
    FFNN.plot_cost_of_last_training()

    y_hat = FFNN.feed_forward(X)
    print('PREDICTED')
    print(y_hat)
    print('ACTUAL')
    print(y)


if __name__ == "__main__":
    x_1, x_2, y = helper.generate_data(100, noise_multiplier=0.01)
    X = np.array(list(zip(x_1, x_2)))
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    main3(X_train, X_test, y_train, y_test)
