import numpy as np
from gradient_descent import SGD

class Neural_Network():
    def __init__(self, input_size, output_size, hidden_size, no_of_hidden):
        """
        Initializing a neural network with the given hyperparameters, 
        creating initial weights according to the standard 
        normal distribution

        :param input_size (int):
            the number of input variables
        :param output_size (int):
            the number of input variables
        :param hidden_size (int):
            the number of hidden layers
        :param no_of_hidden (int):
            the number of hidden layers
        :param bias (float):
            bias in each hidden layer 

        """

        # Setting up the Hyperparameters
        self.input_layer_size = input_size
        self.output_layer_size = output_size
        self.hidden_layer_size = hidden_size
        self.number_of_hidden_layers = no_of_hidden

        # Initialize weights, bias using standard normal distribution
        self.bias = np.random.randn(self.number_of_hidden_layers)
        self.weights = np.zeros((self.number_of_hidden_layers, self.input_layer_size, self.hidden_layer_size))
        for i in range(self.number_of_hidden_layers):
            self.weights[i] = np.random.randn(
                        self.input_layer_size, 
                        self.hidden_layer_size
                    )
            

        # TODO: hidden_to_hidden

        # Other weights, bias for the last 
        self.weights_output = np.random.randn(
                    self.hidden_layer_size, 
                    self.output_layer_size
                )
    
    def feed_forward(self, X, weights = None):
        """
        Running through our network with the starting input values
        of X

        :param X (np.ndarray):
            input variables

        :return (np.ndarray):
            the predicted value given our network
        """

        # Iterate through the hidden layers

        if f'{type(weights)}' == "<class 'NoneType'>":
            weights = self.weights
        else:   
            print('using input weights inside feed-forward')

        print(X.shape) 
        print(weights.shape) 
        print(weights[0].shape) 
        print(np.array([[3, 2, 1]]).shape)
        wawa = weights 
        cal = X @ wawa @ np.array([[3], [1]])

        no_data = X.shape[0]
        
        # z = (no_hidden layers, no_data, hidden_øayer_size)
        self.z = np.zeros((self.number_of_hidden_layers, no_data, self.hidden_layer_size))
        self.a = np.zeros((self.number_of_hidden_layers, no_data, self.hidden_layer_size))
   

        for i in range(self.number_of_hidden_layers):
            cal = X @ weights[i] + self.bias[i]
            # TODO: More hidden layers, must switch X with z[i-1]
            if isinstance(cal, np.ndarray):
                self.z[i] = X @ weights[i] + self.bias[i]
            else: 
                self.z[i] = (X @ weights[i] + self.bias[i])._value

            self.a[i] = self.sigmoid(self.z[i])

        # Propagate to the output layer
        cal_output = self.a[-1] @ self.weights_output + self.bias[-1]
        if isinstance(cal_output, np.ndarray):
            self.z_output = cal_output
        else: 
            self.z_output = cal_output._value

        y_hat = self.sigmoid(self.z_output)

        ## TODO: Last activation is softmax
        return y_hat


    def back_propagation(self, X, y):
        
        start_weights = self.weights
      
        weights, num_of_iter = SGD(start_weights, 
            eta = 0.1, 
            C = self.cost_function_NN, 
            n_epochs=10, 
            M=2, 
            X=X, 
            y=y)

        self.weights = weights

    def lett(self, lol, weights):
        return weights + self.number_of_hidden_layers

    def cost_function_NN(self, weights, X, y, lmbda):
        # TODO: remove lambda
        # ?? Dette kan være feil
        print('STARTT 11111111111111111111111')
        print(weights.__hash__)
        # ?? feed_forward, egen? Som tar inn weights
        print('-letetet')
        print(X)
        print(weights)
        y_hat = self.feed_forward(X, weights)
        print('y_hat:')
        print(y_hat)
        # TODO: cost_function outside of NN
        # print('start cost')
        # print(weights)
        # return self.lett(5, weights)
        return 1/2 * np.mean((y - y_hat)**2)

    def sigmoid(self, x):
        """
        Apply the sigmoid activation function to
        scalar, vectors or matrices

        :param x (float):
            input value

        :return (float):
            the function value
        """

        return 1/(1 + np.exp(-x))

    def sigmoid_prime(self, x):
        # Derivative of Sigmoid Function
        return np.exp(-x)/((1 + np.exp(-x))**2)

 

def main():

    # TODO: scaling of data
    FFNN = Neural_Network(2, 1, 2, 1)
    # Data matrix is (number pf data points, number of data variables)
    X = np.array([[14, 25], [13, 10], [0,0]])
    # ?? kan være 1d array her
    y = np.array([[1], [2], [3]])

    y_hat_start = FFNN.feed_forward(X)
    FFNN.back_propagation(X, y)
    y_hat_stop = FFNN.feed_forward(X)
    print('-----')
    print(f'y_hat_start \n{y_hat_start}')
    print(f'y_hat_stop \n{y_hat_stop}')
    print(f'y_hat_fasit \n{y}')
    print('-------')


    

if __name__ == "__main__":
    main()

