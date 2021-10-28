import numpy as np
# from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
import helper
np.seterr(all='warn')
np.seterr(over='raise')


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
        self.hidden_bias = np.zeros(
            (self.number_of_hidden_layers, self.hidden_layer_size)) + 0.01

        # TODO: works for different output_layer_size?
        self.output_bias = 0.01

        self.input_weights = np.random.randn(
            self.input_layer_size, self.hidden_layer_size)

        if self.number_of_hidden_layers > 1:
            self.hidden_weights = np.zeros(
                (self.number_of_hidden_layers - 1, self.hidden_layer_size, self.hidden_layer_size))
            for i in range(self.number_of_hidden_layers - 1):
                self.hidden_weights[i] = np.random.randn(
                    self.hidden_layer_size,
                    self.hidden_layer_size
                )
        else:
            self.hidden_weights = 0
        # Other weights, bias for the last
        self.output_weights = np.random.randn(
            self.hidden_layer_size, self.output_layer_size)

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
        no_data = X.shape[0]

        # z = (no_hidden layers, no_data, hidden_øayer_size)
        self.z = np.zeros((self.number_of_hidden_layers,
                          no_data, self.hidden_layer_size))
        self.a = np.zeros((self.number_of_hidden_layers,
                          no_data, self.hidden_layer_size))

        for i in range(self.number_of_hidden_layers):
            if i == 0:
                self.z[i] = X @ self.input_weights + self.hidden_bias[i]
            else:
                self.z[i] = self.a[i-1] @ self.hidden_weights[i-1] + \
                    self.hidden_bias[i]

            self.a[i] = self.sigmoid(self.z[i])

        # Propagate to the output layer
        self.z_output = self.a[-1] @ self.output_weights + self.output_bias

        # TODO: Last activation is softmax
        y_hat = self.sigmoid(self.z_output)

        return y_hat

    def train_model(self, X, y, eta=0.05, n_epochs=1000, M=3, gamma=0.5,  cost_method='OLS', lmbda=0.1):

        self.SGD(
            X=X,
            y=y,
            eta=eta,
            n_epochs=n_epochs,
            M=M,
            gamma=gamma,
            cost_method=cost_method,
            lmbda=lmbda
        )

    def backpropagation(self, X, y, cost_method='OLS', lmbda=0.1):
        """
        # TODO:
        """

        # backward propogate through the network

        y_hat = self.feed_forward(X)

        # TODO: deriverte softmax??
        if cost_method == 'OLS':
            output_error = y - y_hat  # error in output
        elif cost_method == 'RIDGE':
            output_error = y - y_hat + np.mean  # TODO RIDGE

        output_delta = self.sigmoid(y_hat, deriv=True) * output_error
        output_weights_grad = - self.a[-1].T @ output_delta
        output_bias_grad = - np.mean(output_delta, axis=0)
        hidden_bias_grad = np.zeros(
            (self.number_of_hidden_layers, self.hidden_layer_size))

        if self.number_of_hidden_layers > 1:
            hidden_weights_grad = np.zeros(
                (self.number_of_hidden_layers - 1, self.hidden_layer_size, self.hidden_layer_size))
            hidden_error = output_delta @ self.output_weights.T
            hidden_delta = self.sigmoid(self.z[-1], deriv=True) * hidden_error
            hidden_weights_grad[-1] = - self.a[-2].T @ output_delta

            for i in range(self.number_of_hidden_layers - 2):
                # endret
                hidden_error = hidden_delta @ self.hidden_weights[-(i+1)].T
                hidden_delta = self.sigmoid(
                    self.z[-(i+2)], deriv=True) * hidden_error
                hidden_weights_grad[-(i+2)] = - self.a[-(i+3)].T @ hidden_delta
                hidden_bias_grad[-(i+2)] = - np.mean(hidden_delta, axis=0)

            input_error = hidden_delta @ self.hidden_weights[0].T

        elif self.number_of_hidden_layers == 1:
            input_error = output_delta @ self.output_weights.T
            hidden_weights_grad = 0

        input_delta = self.sigmoid(self.z[0], deriv=True) * input_error

        hidden_bias_grad[0] = - np.mean(input_delta, axis=0)
        input_weights_grad = - X.T @ input_delta

        return input_weights_grad, hidden_weights_grad, output_weights_grad, hidden_bias_grad, output_bias_grad

    def SGD(self, X, y, eta, n_epochs, M, gamma=0, cost_method='OLS', lmbda=0.1, tol=1e-14):
        """
        # TODO:
        """

        input_weights_previous = self.input_weights
        hidden_weights_previous = self.hidden_weights
        output_weights_previous = self.output_weights
        hidden_bias_previous = self.hidden_bias
        output_bias_previous = self.output_bias

        n = X.shape[0]
        # TODO: change
        m = int(n/M)

        v_input_weight = 0
        v_hidden_weight = 0
        v_output_weight = 0
        v_hidden_bias = 0
        v_output_bias = 0

        j = 0
        error_list = []

        for epoch in range(n_epochs):
            for i in range(m):
                # Do something with the end interval of the selected
                k = np.random.randint(m)

                # TODO: random integer can be wrongFinding the k-th batch
                xk_batch = X[k*M:(k+1)*M]
                yk_batch = y[k*M:(k+1)*M]

                # hidden_grad = self.backpropagation(xk_batch, yk_batch) #weights are updated
                input_weights_grad, hidden_weights_grad, output_weights_grad, \
                    hidden_bias_grad, output_bias_grad = self.backpropagation(
                        xk_batch, yk_batch, cost_method, lmbda)  # weights are updated

                v_input_weight = gamma*v_input_weight + eta*input_weights_grad
                input_weights_next = input_weights_previous - v_input_weight

                v_hidden_weight = gamma*v_hidden_weight + eta*hidden_weights_grad
                hidden_weights_next = hidden_weights_previous - v_hidden_weight

                v_output_weight = gamma*v_output_weight + eta*output_weights_grad
                output_weights_next = output_weights_previous - v_output_weight

                v_hidden_bias = gamma*v_hidden_bias + eta*hidden_bias_grad
                hidden_bias_next = hidden_bias_previous - v_hidden_bias

                v_output_bias = gamma*v_output_bias + eta*output_bias_grad
                output_bias_next = output_bias_previous - v_output_bias

                # TODO: updating eta?

                j += 1
                # Check if we have reached the tolerance
                # if np.sum(np.abs(theta_next - theta_previous)) < tol:
                #     print('local')
                #     return theta_next, j

                # Updating the thetas
                input_weights_previous = input_weights_next
                hidden_weights_previous = hidden_weights_next
                output_weights_previous = output_weights_next
                hidden_bias_previous = hidden_bias_next
                output_bias_previous = output_bias_next

                error_list.append(np.mean((self.feed_forward(X) - y)**2))

                self.input_weights = input_weights_next
                self.hidden_weights = hidden_weights_next
                self.output_weights = output_weights_next
                self.hidden_bias = hidden_bias_next
                self.output_bias = output_bias_next

        plt.loglog(range(len(error_list)), error_list)
        plt.show()

    def sigmoid(self, x, deriv=False):
        """
        Apply the sigmoid activation function to
        scalar, vectors or matrices

        :param x (float):
            input value

        :return (float):
            the function value
        """
        if deriv:
            try:
                ret = np.exp(-x)/((1+np.exp(-x))**2)
            except Exception or Warning as e:
                # Refreshing the variables
                print(f'>> ERROR: {e}')
                print('maybe turn down the complexity')
                exit()
        else:
            try:
                ret = 1/(1 + np.exp(-x))
            except Exception or Warning as e:
                # Refreshing the variables
                print(f'>> ERROR {e}')
                print('maybe reduce the complexity')
                exit()

        return ret


def main():
    print("-----STARTING MAIN -----")
    # TODO: scaling of data
    FFNN = Neural_Network(2, 1, 20, 3)

    # Generate some data from the Franke Function
    x_1, x_2, y = helper.generate_data(100, noise_multiplier=0)
    X = np.array(list(zip(x_1, x_2)))
    y = y.reshape(-1, 1)
    y_scalar = max(y)
    y /= y_scalar

    # Splitting the data in train and testing
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    FFNN.train_model(
        X_train, y_train,
        eta=0.05, n_epochs=5000,
        M=3, gamma=0.8)

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

    # RIDGE:
    y_hat_test_RIDGE, y_hat_train_RIDGE, _ = helper.predict_output(
        x_train=X_train[:, 0], y_train=X_train[:, 1], z_train=y_train,
        x_test=X_test[:, 0], y_test=X_test[:, 1],
        degree=4, regression_method='RIDGE', lmbda=0.01
    )

    print('\n\n\n>>> CHECKING OUR MODEL: \n')
    print('Neural Network:')
    print('-- (MSE) TRAINING DATA --')
    print(helper.mean_squared_error(y_train, y_hat_train))
    print('-- (MSE) TESTING DATA --')
    print(helper.mean_squared_error(y_test, y_hat_test))
    print(f'-- (R2) TESTING DATA --')
    print(helper.r2_score(y_hat_test, y_test))

    print('\nOLS:')
    print('-- TESTING DATA --')
    print(helper.mean_squared_error(y_train, y_hat_train_OLS))
    print('-- TRAINING DATA: --')
    print(helper.mean_squared_error(y_test, y_hat_test_OLS))
    print(f'-- (R2) TESTING DATA --')
    print(helper.r2_score(y_hat_test_OLS, y_test))

    print('\nRIDGE:')
    print('-------TESTING DATA-------')
    print(helper.mean_squared_error(y_train, y_hat_train_RIDGE))
    print('-------TRAINING DATA:-------')
    print(helper.mean_squared_error(y_test, y_hat_test_RIDGE))
    print(f'-- (R2) TESTING DATA --')
    print(helper.r2_score(y_hat_test_RIDGE, y_test))
    print('New')


if __name__ == "__main__":
    main()
