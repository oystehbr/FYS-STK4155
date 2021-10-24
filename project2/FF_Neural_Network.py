import autograd.numpy as np
from autograd import elementwise_grad as egrad

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
        self.bias = np.zeros(self.number_of_hidden_layers)    # self.bias = np.random.randn(self.number_of_hidden_layers)
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
    
    def feed_forward(self, X):
        """
        Running through our network with the starting input values
        of X

        :param X (np.ndarray):
            input variables

        :return (np.ndarray):
            the predicted value given our network
        """

        # Iterate through the hidden layers

        weights = self.weights

        no_data = X.shape[0]
        
        # z = (no_hidden layers, no_data, hidden_øayer_size)
        self.z = np.zeros((self.number_of_hidden_layers, no_data, self.hidden_layer_size))
        self.a = np.zeros((self.number_of_hidden_layers, no_data, self.hidden_layer_size))


        for i in range(self.number_of_hidden_layers):
            # TODO: More hidden layers, must switch X with z[i-1]
            self.z[i] = X @ weights[i] + self.bias[i]
            self.a[i] = self.sigmoid(self.z[i])

        # Propagate to the output layer
        self.z_output = self.a[-1] @ self.weights_output + self.bias[-1]

        y_hat = self.sigmoid(self.z_output)

        ## TODO: Last activation is softmax
        return y_hat


    def train_model(self, X, y):
      
        self.SGD(
            X=X,
            y=y,
            eta = 0.1,  
            n_epochs=10000, 
            M=2, 
            gamma=0.5
            )



    def train(self, X, y):
        output = self.feed_forward(X)
        self.backward(X, y, output)


    def backpropagation(self, X, y):
        #TODO: make thid work for many hidden layers
        #backward propogate through the network
        # y = actual, output = predicted
        y_hat = self.feed_forward(X)
        print("Error:", end = "    ")
        print(np.mean(y_hat - y))
        
        self.output_error = y - y_hat  # error in output
        self.output_delta = self.output_error * self.sigmoid(y_hat, deriv=True)

        # For-loop
        # Den må være 1*3
        self.z2_error = self.output_error @ self.weights_output.T    # z2 error: how much our hidden layer weights contribute to the output error
        self.z2_delta = self.z2_error * self.sigmoid(self.z[0], deriv=True)   # applying derivative of sigmoid to z2 error

        hidden_weights_grad = - X.T.dot(self.z2_delta) 
        output_weights_grad = - self.z[0].T @ self.output_delta

        return hidden_weights_grad, output_weights_grad
        # self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input -> hidden layer)
        # self.W2 += self.z2.T.dot(self.output_delta) # adjusting the second set (hidden -> output layer)


    def SGD(self, X, y, eta, n_epochs, M, gamma=0, tol=1e-14):
        """
        # TODO:
        """

        hidden_weights_previous = self.weights
        output_weights_previous = self.weights_output

        def learning_schedule(t):
            return 5/(t+50)

        n = X.shape[0]
        # TODO: change
        # m = int(n/M)
        m = 5

        v_hidden_weight = 0
        v_output_weight = 0

        # TODO: updating v, will be wrong if no GD, make class Brolmsen
        
        j = 0
        for epoch in range(n_epochs): 
            for i in range(m):
                # Do something with the end interval of the selected
                k = np.random.randint(m)

                # Finding the k-th batch
                xk_batch = X[k*M:(k+1)*M]
                yk_batch = y[k*M:(k+1)*M]
                
                # hidden_grad = self.backpropagation(xk_batch, yk_batch) #weights are updated
                hidden_weights_grad, output_weights_grad = self.backpropagation(X, y) #weights are updated

                v_hidden_weight = gamma*v_hidden_weight + eta*hidden_weights_grad
                hidden_weights_next = hidden_weights_previous - v_hidden_weight

                v_output_weight = gamma*v_output_weight + eta*output_weights_grad
                output_weights_next = output_weights_previous - v_output_weight
                
                # TODO: add this one
                # eta = learning_schedule(epoch*i*m)
            
                j += 1
                # Check if we have reached the tolerance
                # if np.sum(np.abs(theta_next - theta_previous)) < tol:
                #     print('local')
                #     return theta_next, j

                # Updating the thetas
                output_weights_previous = output_weights_next
                hidden_weights_previous = hidden_weights_next

                self.weights[0] = hidden_weights_next
                self.weights_output = output_weights_next


    def sigmoid(self, x, deriv = False):
        """
        Apply the sigmoid activation function to
        scalar, vectors or matrices

        :param x (float):
            input value

        :return (float):
            the function value
        """
        if deriv:
            return np.exp(-x)/((1+np.exp(-x))**2)
            # return x * (1 - x)
        else: 
            return 1/(1 + np.exp(-x))


def main():
    X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
    y = np.array(([92], [86], [89]), dtype=float)
    # TODO: scaling of data
    FFNN = Neural_Network(2, 1, 200, 1)
    # Data matrix is (number pf data points, number of data variables)
    X = np.array(([0, 0], [1, 1], [4, 4])) #, [3, 3], [4, 4]))#, [5, 5], [6, 6], [7, 7]))
    # ?? kan være 1d array her
    y = np.array(([0], [1], [16])) #, [30])) #, [80])) #, [25], [36], [49]))
    X = X/np.amax(X, axis = 0)
    y = y/100
    # X = X[:-1]
    # y = y[:-1]
    # X_pred = X[-1]
    # y_pred = y[-1]


    print("----START: ----")
    print(FFNN.weights)
    print(FFNN.feed_forward(X))
    FFNN.train_model(X, y)
    print('----AFTER----')
    print(FFNN.weights)
    print(FFNN.feed_forward(X))

    print('FAKTISK:')
    print(y)


    # print('OUT OF SAMPLE:')
    # print(FFNN.feed_forward(X_pred))
    # print(y_pred)




    
if __name__ == "__main__":
    main()


"""
UserWarning: Output seems independent of input.
  warnings.warn("Output seems independent of input.")
"""