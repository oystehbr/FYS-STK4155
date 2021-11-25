"""
Setup for testing the results of project 2, feel free to change the
different variables. Read docstrings inside the functions/ classes
to know their functionality.

We have provided some explanation of the different tests above the test,
you need to set the test you want to run to True.

So, if you wanna run TEST 1, change the variable test1:
test1 = True

To use this file efficiently:
VSCODE: CTRL K -> CTRL 0 (close all if-statements, functions etc.)
    -> find the similar thing in other editors by doing some small research at google
then this file will be very easy to read/ use
"""

from FF_Neural_Network import Neural_Network
from gradient_descent import SGD
from cost_functions import logistic_cost_NN, cost_logistic_regression, prob
import time
import numpy as np
import gradient_descent
import helper
import seaborn as sns

"""
TEST 1
DATASET: FRANKE FUNCTION (regression case)

Analysis of the gradient descent algorithm, and comparing it
against the results of project 1. For the Ridge case, it will provide
a seaborn visualization with hyperparameter lambda and the learning rate
"""

test1 = False
if test1:
    print('>> RUNNING TEST 1:')
    # Generating some data (Franke Function)
    n = 500
    noise = 0.01
    x_values, y_values, z_values = helper.generate_data(n, noise)
    number_of_epochs = 200

    degree = 4  # complexity of the model
    gamma = 0.7  # the momentum of the stochastic gradient decent

    "Set to true, stochastic gradient decent testing with OLS"
    run_main_OLS = False
    if run_main_OLS:
        print('> Analysing the gradient descent algorithm with OLS')

        # Set the number of minibatches you want to analyse
        list_number_of_minibatches = [1, 5, 8, 10, 16, 20, 40, n*0.8]
        gradient_descent.main_OLS(
            x_values=x_values, y_values=y_values, z_values=z_values,
            list_no_of_minibatches=list_number_of_minibatches,
            n_epochs=number_of_epochs,
            degree=degree, gamma=gamma
        )

    no_of_minibatches = 20
    "Set to true, stochastic gradient decent testing with RIDGE"
    run_main_RIDGE = True
    if run_main_RIDGE:
        print('> Analysing the gradient descent algorithm with RIDGE')
        gradient_descent.main_RIDGE(
            x_values=x_values, y_values=y_values, z_values=z_values,
            no_of_minibatches=no_of_minibatches,
            n_epochs=number_of_epochs,
            degree=degree, gamma=gamma
        )

    if not run_main_OLS and not run_main_RIDGE:
        print('You need to go into the test and set some booleans to be able to see the testing ')


"""
TEST 2
DATASET: FRANKE FUNCTION (regression case)

Here you are able to try out the neural network and see the result in the form of 
R2-score (of both training and test data). We have also provided the 
opportunity to look at the cost-value over the training time (iterations of the 
SGD-algorithm).
"""

test2 = False
if test2:
    print('>> RUNNING TEST 2:')
    # Initializing some data
    n = 100
    noise = 0.01
    x1, x2, y = helper.generate_data(n, noise)
    X = np.array(list(zip(x1, x2)))
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    # Initializing the Neural Network
    node_list = [5]*2

    FFNN = Neural_Network(
        no_input_nodes=2,
        no_output_nodes=1,
        node_list=node_list
    )

    # Setting the preffered activation functions for hidden layer
    FFNN.set_activation_function_hidden_layers('sigmoid')

    # Set the preffered values of the gradient descent
    FFNN.set_SGD_values(
        eta=0.01,
        lmbda=1e-5,
        n_epochs=1000,
        batch_size=10,
        gamma=0.7
    )

    # Now, we are ready to train the model
    keep_cost_values = False
    FFNN.train_model(X_train, y_train, keep_cost_values)

    if keep_cost_values:
        # If True: we can plot the cost vs. iterations
        FFNN.plot_cost_of_last_training()

    # Finding the predicted values with our model
    y_hat_train = FFNN.feed_forward(X_train)
    y_hat_test = FFNN.feed_forward(X_test)

    # for k in range(10):


    # Checking the result
    print('\n\n>>> CHECKING THE MODEL:')
    print('>> (R2) TRAINING DATA: ', end='')
    print(helper.r2_score(y_train, y_hat_train))
    print(f'>> (R2) TESTING DATA: ', end='')
    print(helper.r2_score(y_hat_test, y_test))


"""
TEST 3
DATASET: FRANKE FUNCTION (regression case)

Testing the Neural Network, vs. OLS- and RIDGE-regression. It will
provide the results from the three different methods in R2-score. And the
time spent for creating these models.
"""

test3 = False
if test3:
    print('>> RUNNING TEST 3:')
    # Initializing some data
    n = 100
    noise = 0.01
    x1, x2, y = helper.generate_data(n, noise)
    X = np.array(list(zip(x1, x2)))
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    # Initializing the Neural Network
    neural_start = time.time()
    number_of_input_nodes = 2
    number_of_output_nodes = 1
    node_list = [40]*4

    FFNN = Neural_Network(
        no_input_nodes=2,
        no_output_nodes=1,
        node_list=node_list
    )

    # Setting the preffered activation functions (for hidden and output layer)
    FFNN.set_activation_function_hidden_layers('sigmoid')

    # Set the preffered values of the gradient descent
    FFNN.set_SGD_values(
        eta=1e-4,
        lmbda=1e-4,
        n_epochs=5000,
        batch_size=16,
        gamma=0.5
    )

    # Now, we are ready to train the model
    keep_cost_values = False
    FFNN.train_model(X_train, y_train, keep_cost_values)

    # Finding the predicted values with Neural Network
    y_hat_train = FFNN.feed_forward(X_train)
    y_hat_test = FFNN.feed_forward(X_test)
    neural_time = time.time() - neural_start

    if keep_cost_values:
        # If True: we can plot the cost vs. iterations
        FFNN.plot_cost_of_last_training()

    complexity_degree = 4
    ols_start = time.time()
    # OLS:
    y_hat_test_OLS, y_hat_train_OLS, _ = helper.predict_output(
        x_train=X_train[:, 0], y_train=X_train[:, 1], z_train=y_train,
        x_test=X_test[:, 0], y_test=X_test[:, 1],
        degree=complexity_degree, regression_method='OLS'
    )
    ols_time = time.time() - ols_start

    # RIDGE:
    lmbda = 0.01
    ridge_start = time.time()
    y_hat_test_RIDGE, y_hat_train_RIDGE, _ = helper.predict_output(
        x_train=X_train[:, 0], y_train=X_train[:, 1], z_train=y_train,
        x_test=X_test[:, 0], y_test=X_test[:, 1],
        degree=complexity_degree, regression_method='RIDGE', lmbda=lmbda)
    ridge_time = time.time() - ridge_start

    # Printing out the result for comparision
    print('\n\n>>> CHECKING OUR MODEL: ')
    print(f'\n> Neural Network (time spent: {neural_time:.4f}s):')
    print(f'** (R2) TRAINING DATA: ', end='')
    print(round(helper.r2_score(y_hat_train, y_train), 4))
    print(f'** (R2)  TESTING DATA: ', end='')
    print(round(helper.r2_score(y_hat_test, y_test), 4))

    print(f'\n> OLS (time spent: {ols_time:.4f}s):')
    print(f'** (R2) TRAINING DATA: ', end='')
    print(round(helper.r2_score(y_hat_train_OLS, y_train), 4))
    print(f'** (R2)  TESTING DATA: ', end='')
    print(round(helper.r2_score(y_hat_test_OLS, y_test), 4))

    print(f'\n> RIDGE (time spent: {ridge_time:.4f}s):')
    print(f'** (R2) TRAINING DATA: ', end='')
    print(round(helper.r2_score(y_hat_train_RIDGE, y_train), 4))
    print(f'** (R2)  TESTING DATA: ', end='')
    print(round(helper.r2_score(y_hat_test_RIDGE, y_test), 4))


"""
TEST 4
DATASET: FRANKE FUNCTION (regression case)

Optimizing the hyperparameter lmbda and the learning rate by looking over 
a seaborn plot. Will be measured in R2-score for both training and test-data. 
"""

test4 = False
if test4:
    print('>> RUNNING TEST 4:')
    # Initializing some data
    n = 100
    noise = 0.01
    x1, x2, y = helper.generate_data(n, noise)
    X = np.array(list(zip(x1, x2)))
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    # Initializing the Neural Network
    num_hidden_nodes = 8
    num_hidden_layers = 4
    node_list = [num_hidden_nodes]*num_hidden_layers
    FFNN = Neural_Network(
        no_input_nodes=2,
        no_output_nodes=1,
        node_list=node_list
    )

    # Setting the preffered activation functions for hidden layer
    hidden_layer_name = "Leaky_RELU"
    FFNN.set_activation_function_hidden_layers(hidden_layer_name)

    # Set the preffered values of the gradient descent
    n_epochs = 1000
    batch_size = 5
    gamma = 0.6
    FFNN.set_SGD_values(
        n_epochs=n_epochs,
        batch_size=batch_size,
        gamma=gamma
    )

    # Starting up a seaborn plot
    sns.set()

    # SO PROBABLY CANCEL THE EXIT()
    learning_rates = np.logspace(-3, -6, 4)
    lmbda_values = np.logspace(-3, -6, 4)

    train_R2_score = np.zeros((len(learning_rates), len(lmbda_values)))
    test_R2_score = np.zeros((len(learning_rates), len(lmbda_values)))

    iter = 0
    for i, eta in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbda_values):
            # Refreshing the weights and biases before testing new SGD_values
            FFNN.initialize_the_biases()
            FFNN.initialize_the_weights()

            # Setting the preffered SGD_values
            FFNN.set_SGD_values(
                eta=eta,
                lmbda=lmbda
            )

            FFNN.train_model(X_train, y_train)

            # Finding the predicted values with our model
            y_hat_train = FFNN.feed_forward(X_train)
            y_hat_test = FFNN.feed_forward(X_test)

            train_R2_score[i][j] = helper.r2_score(y_train, y_hat_train)
            test_R2_score[i][j] = helper.r2_score(y_test, y_hat_test)

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(learning_rates) * len(lmbda_values)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_lmbda_learning(
        score=train_R2_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name="Training R2-score",
        save_name=f'plots/test4/test4_M_{batch_size}_gamma_{gamma}_hiddennodes{num_hidden_nodes}_hiddenlayers_{num_hidden_layers}_name_hidden_{hidden_layer_name}_training_2.png'
    )
    helper.seaborn_plot_lmbda_learning(
        score=test_R2_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name="Test R2-score",
        save_name=f'plots/test4/test4_M_{batch_size}_gamma_{gamma}_hiddennodes{num_hidden_nodes}_hiddenlayers_{num_hidden_layers}_name_hidden_{hidden_layer_name}_test_2.png'
    )


"""
TEST 5:
DATASET: CANCER DATA (classification case)

Here you are able to try out the neural network and see the result in the form of 
an accuracy score (of both training and test data). We have also provided the 
opportunity to look at the accuracy score over the training time (iterations of the 
SGD-algorithm).
"""

test5 = True
if test5:
    print('>> RUNNING TEST 5:')
    # Loading the training and testing dataset
    n_components = 2
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = helper.load_cancer_data(
        n_components)

    # Setting the architecture of the Neural Network
    node_list = [20]*3

    # Initializing the Neural Network
    FFNN = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=1,
        node_list=node_list
    )

    # Setting the preffered Stochastic Gradient Descent parameters
    FFNN.set_SGD_values(
        n_epochs=60,
        batch_size=14,
        gamma=0.5,
        eta=1e-3,
        lmbda=1e-5)

    # Setting the preffered cost- and activation functions
    FFNN.set_cost_function(logistic_cost_NN)
    FFNN.set_activation_function_hidden_layers('sigmoid')
    FFNN.set_activation_function_output_layer('sigmoid')

    # Set keep_accuracy to True, if you wanna see accuracy vs. time (in the training)
    keep_accuracy = True
    FFNN.train_model(X_cancer_train, y_cancer_train,
                     keep_accuracy_score=keep_accuracy)

    y_hat = FFNN.feed_forward(X_cancer_train)

    for k in range(10):
        print(y_cancer_train[k])
        print(y_hat[k])

    # Change the activation function to predict 0 or 1's.
    FFNN.set_activation_function_output_layer('sigmoid_classification')
    print(
        f'Accuracy_train = {helper.accuracy_score(FFNN.feed_forward(X_cancer_train),  y_cancer_train)}')
    print(
        f'Accuracy_test = {helper.accuracy_score(FFNN.feed_forward(X_cancer_test),  y_cancer_test)}')

    if keep_accuracy:
        FFNN.plot_accuracy_score_last_training()

"""
TEST 6:
DATASET: CANCER DATA (classification case)

Optimizing the hyperparameter lmbda and the learning rate by looking over 
a seaborn plot. Will be measured in accuracy-score for both training and test-data. 
"""

test6 = False
if test6:
    print('>> RUNNING TEST 6:')
    # Loading the training and testing dataset
    n_components = 2
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = helper.load_cancer_data(
        n_components)

    # Setting the architecture of the Neural Network
    no_hidden_nodes = 20
    no_hidden_layers = 1
    node_list = [no_hidden_nodes]*no_hidden_layers

    # Initializing the Neural Network
    FFNN = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=1,
        node_list=node_list
    )

    # Setting the preffered cost- and hidden_activation function
    FFNN.set_cost_function(logistic_cost_NN)
    FFNN.set_activation_function_hidden_layers('sigmoid')

    # Change the activation function to predict 0 or 1's.
    learning_rates = [0.0009, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.00005]
    lmbda_values = np.logspace(-2, -7, 6)

    train_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))
    test_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))

    n_epochs = 200
    batch_size = 14
    gamma = 0.9

    FFNN.set_SGD_values(
        n_epochs=n_epochs,
        batch_size=batch_size,
        gamma=gamma)

    iter = 0
    for i, eta in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbda_values):
            # Reinitializing the weights, biases and activation function
            output_activation = 'sigmoid'
            FFNN.set_activation_function_output_layer(output_activation)
            FFNN.initialize_the_weights()
            FFNN.initialize_the_biases()

            # Changing some SGD values
            FFNN.set_SGD_values(
                eta=eta,
                lmbda=lmbda,
            )

            # Training the model
            FFNN.train_model(X_cancer_train, y_cancer_train)

            # Testing the model against the target values, and store the results
            FFNN.set_activation_function_output_layer(
                'sigmoid_classification')
            train_accuracy_score[i][j] = helper.accuracy_score(
                y_cancer_train, FFNN.feed_forward(X_cancer_train))
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_cancer_test, FFNN.feed_forward(X_cancer_test))

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(learning_rates) * len(lmbda_values)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_lmbda_learning(
        score=train_accuracy_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Training Accuracy',
        save_name=f'plots/test6/test6_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_features_{n_components}_hiddennodes_{no_hidden_nodes}_hiddenlayer_{no_hidden_layers}_actOUT_{output_activation}_training_14.png'
    )

    helper.seaborn_plot_lmbda_learning(
        score=test_accuracy_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Test Accuracy',
        save_name=f'plots/test6/test6_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_features_{n_components}_hiddennodes_{no_hidden_nodes}_hiddenlayer_{no_hidden_layers}_actOUT_{output_activation}_test_14.png'
    )


"""
TEST 7:
DATASET: CANCER DATA (classification case)

Logistic regression: create a seaborn plot of the hyperparameter lambda
and the learning rate. Other SGD values, will be easy to change in the test.
"""
test7 = False
if test7:
    print('>> RUNNING TEST 7:')
    # Loading the training and testing dataset
    n_components = 2
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = helper.load_cancer_data(
        n_components)

    learning_rates = np.logspace(1, -3, 5)
    lmbda_values = np.logspace(-1, -7, 7)

    train_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))
    test_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))

    n_epochs = 40
    batch_size = 14
    gamma = 0.8
    iter = 0
    for i, eta in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbda_values):
            theta, num = SGD(
                X=X_cancer_train, y=y_cancer_train,
                theta_init=np.array([0.1]*(X_cancer_train.shape[1] + 1)),
                eta=eta,
                cost_function=cost_logistic_regression,
                n_epochs=n_epochs, batch_size=batch_size,
                gamma=gamma,
                lmbda=lmbda
            )

            # Finding the predicted values
            predicted_values_train = np.where(
                prob(theta, X_cancer_train) >= 0.5, 1, 0)
            predicted_values_test = np.where(
                prob(theta, X_cancer_test) >= 0.5, 1, 0)

            # Applying the model against the target values, and store the results
            train_accuracy_score[i][j] = helper.accuracy_score(
                y_cancer_train, predicted_values_train)
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_cancer_test, predicted_values_test)

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(learning_rates) * len(lmbda_values)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_lmbda_learning(
        score=train_accuracy_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Training Accuracy',
        save_name=f'plots/test7/test7_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_features_{n_components}_training.png'
    )
    helper.seaborn_plot_lmbda_learning(
        score=test_accuracy_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Test Accuracy',
        save_name=f'plots/test7/test7_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_features_{n_components}_test.png'
    )


"""
TEST 8:
DATASET: CANCER DATA (classification case)

Comparison: logistic regression vs classification with Neural Network. Comparing
the R2-score on testing and training data, and the time spent on training the model.
"""
test8 = False
if test8:
    print('>> RUNNING TEST 8:')
    "First, setting up the Neural Network and finding the results"

    # Loading the training and testing dataset
    n_components = 2
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = helper.load_cancer_data(
        n_components)

    time_NN_start = time.time()
    # Setting the architecture of the Neural Network
    node_list = [20]*1

    # Initializing the Neural Network
    FFNN = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=1,
        node_list=node_list,
    )

    # Setting the preffered Stochastic Gradient Descent parameters
    FFNN.set_SGD_values(
        n_epochs=4,
        batch_size=14,
        gamma=0.5,
        eta=0.001,
        lmbda=1e-5)

    # Setting the preffered cost- and activation functions
    FFNN.set_cost_function(logistic_cost_NN)
    FFNN.set_activation_function_hidden_layers('RELU')
    FFNN.set_activation_function_output_layer('sigmoid')
    FFNN.train_model(X_cancer_train, y_cancer_train)

    # Change the activation function to predict 0 or 1's.
    FFNN.set_activation_function_output_layer('sigmoid_classification')
    print('>>> Results of classification problem with Neural Network <<<')
    print(
        f'ACCURACY_train = {helper.accuracy_score(FFNN.feed_forward(X_cancer_train),  y_cancer_train): .4f}')
    print(
        f'ACCURACY_test = {helper.accuracy_score(FFNN.feed_forward(X_cancer_test),  y_cancer_test): .4f}')
    print(f'TIME SPENT: {time.time() - time_NN_start : .3f}')

    "Secondly, look at the logistic regression and find the results"
    time_logistic_start = time.time()

    theta, num = SGD(
        X=X_cancer_train, y=y_cancer_train,
        # initial guess of the betas
        theta_init=np.array([0.1]*(X_cancer_train.shape[1] + 1)),
        eta=0.1,
        cost_function=cost_logistic_regression,
        n_epochs=3, batch_size=14,
        gamma=0.8,
        lmbda=1e-3
    )

    # Finding the predicted values
    predicted_values_train = np.where(prob(theta, X_cancer_train) >= 0.5, 1, 0)
    predicted_values_test = np.where(prob(theta, X_cancer_test) >= 0.5, 1, 0)

    print('\n>>> Results of logistic regression, with SGD <<<')
    print(
        f'ACCURACY_train => {helper.accuracy_score(predicted_values_train, y_cancer_train): .4f}')
    print(
        f'ACCURACY_test => {helper.accuracy_score(predicted_values_test, y_cancer_test): .4f}')
    print(f'TIME SPENT: {time.time() - time_logistic_start: .3f}')


"""
TEST 9
DATASET: FRANKE FUNCTION (regression case)

Optimizing the architecture of the Neural Network (amount of hidden nodes and layers)
by looking over a seaborn plot. Will be measured in R2-score for both training and test-data. 
"""
test9 = False
if test9:
    print('>> RUNNING TEST 9:')
    # Initializing some data
    n = 100
    noise = 0.01
    x1, x2, y = helper.generate_data(n, noise)
    X = np.array(list(zip(x1, x2)))
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    sns.set()

    # Set the parameters used in the Neural Network
    n_epochs = 1000
    batch_size = 10
    gamma = 0
    eta = 1e-5
    lmbda = 0.0001

    nodes = np.arange(1, 13, 2)
    layers = np.arange(2, 7, 2)
    train_R2_score = np.zeros((len(nodes), len(layers)))
    test_R2_score = np.zeros((len(nodes), len(layers)))
    iter = 0
    for i, node in enumerate(nodes):
        for j, layer in enumerate(layers):
            print(node, layer)
            # Need to create new instance, to change the architecture
            node_list = [node]*layer
            FFNN = Neural_Network(
                no_input_nodes=2,
                no_output_nodes=1,
                node_list=node_list
            )

            # Setting the preffered activation functions for hidden layer
            hidden_layer_name = 'Leaky_RELU'
            FFNN.set_activation_function_hidden_layers(hidden_layer_name)

            # Set the preffered values of the gradient descent
            FFNN.set_SGD_values(
                n_epochs=n_epochs,
                batch_size=batch_size,
                gamma=gamma,
                eta=eta,
                lmbda=lmbda
            )

            FFNN.train_model(X_train, y_train)

            # Finding the predicted values with our model
            y_hat_train = FFNN.feed_forward(X_train)
            y_hat_test = FFNN.feed_forward(X_test)

            train_R2_score[i][j] = helper.r2_score(y_train, y_hat_train)
            test_R2_score[i][j] = helper.r2_score(y_test, y_hat_test)

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(nodes) * len(layers)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_architecture(
        score=train_R2_score,
        x_tics=layers,
        y_tics=nodes,
        score_name='Training R2-score',
        save_name=f'plots/test9/test9_M_{batch_size}_gamma_{gamma}_lmbda_{lmbda}_eta_{eta}_training_layerName{hidden_layer_name}_1.png'
    )
    helper.seaborn_plot_architecture(
        score=test_R2_score,
        x_tics=layers,
        y_tics=nodes,
        score_name='Test R2-score',
        save_name=f'plots/test9/test9_M_{batch_size}_gamma_{gamma}_lmbda_{lmbda}_eta_{eta}_test_layerName{hidden_layer_name}_1.png'
    )

"""
TEST 10
DATASET: FRANKE FUNCTION (regression case)

Optimizing the batch_sizes and the momentum parameter gamma by looking over a seaborn plot. 
Will be measured in R2-score for both training and test-data. 
"""
test10 = False
if test10:
    print('>> RUNNING TEST 10:')
    # Initializing the data
    n = 100
    noise = 0.01
    x1, x2, y = helper.generate_data(n, noise)
    X = np.array(list(zip(x1, x2)))
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    # Setting up the Neural Network
    num_hidden_nodes = 8
    num_hidden_layers = 4
    node_list = [num_hidden_nodes]*num_hidden_layers
    FFNN = Neural_Network(
        no_input_nodes=2,
        no_output_nodes=1,
        node_list=node_list
    )

    # Set the parameters used in the Neural Network
    n_epochs = 800
    eta = 0.0001
    lmbda = 0.0001
    FFNN.set_SGD_values(
        n_epochs=n_epochs,
        eta=eta,
        lmbda=lmbda
    )

    batch_sizes = np.arange(2, 10, 2)
    gammas = [0.1, 0.2, 0.4, 0.6, 0.8]
    train_R2_score = np.zeros((len(batch_sizes), len(gammas)))
    test_R2_score = np.zeros((len(batch_sizes), len(gammas)))
    iter = 0
    for i, batch_size in enumerate(batch_sizes):
        for j, gamma in enumerate(gammas):
            # Refreshing the weights and biases before testing new SGD_values
            FFNN.initialize_the_biases()
            FFNN.initialize_the_weights()

            # Setting the preffered activation functions for hidden layer
            hidden_layer_name = 'Leaky_RELU'
            FFNN.set_activation_function_hidden_layers(hidden_layer_name)

            # Set the preffered values of the gradient descent
            FFNN.set_SGD_values(
                batch_size=batch_size,
                gamma=gamma,
            )

            FFNN.train_model(X_train, y_train)

            # Finding the predicted values with our model
            y_hat_train = FFNN.feed_forward(X_train)
            y_hat_test = FFNN.feed_forward(X_test)

            train_R2_score[i][j] = helper.r2_score(y_train, y_hat_train)
            test_R2_score[i][j] = helper.r2_score(y_test, y_hat_test)

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(batch_sizes) * len(gammas)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_batchsize_gamma(
        score=train_R2_score,
        x_tics=gammas,
        y_tics=batch_sizes,
        score_name='Training R2-score',
        save_name=f'plots/test10/test10_lmbda_{lmbda}_eta_{eta}_hiddennodes_{num_hidden_nodes}_hiddenlayer_{num_hidden_layers}_hidden_layer_name_{hidden_layer_name}_training_2.png'
    )
    helper.seaborn_plot_batchsize_gamma(
        score=test_R2_score,
        x_tics=gammas,
        y_tics=batch_sizes,
        score_name='Test R2-score',
        save_name=f'plots/test10/test10_lmbda_{lmbda}_eta_{eta}_{num_hidden_nodes}_hiddenlayer_{num_hidden_layers}_hidden_layer_name_{hidden_layer_name}_test_2.png'
    )


"""
TEST 11
DATASET: CANCER DATA (classification case)

Optimizing the architecture of the Neural Network (amount of hidden nodes and layers)
by looking over a seaborn plot. Will be measured in accuracy-score for both training and test-data. 
"""
test11 = False
if test11:
    print('>> RUNNING TEST 11:')
    # Loading the training and testing dataset
    n_components = 2
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = helper.load_cancer_data(
        n_components)
    sns.set()

    # Set the parameters used in the Neural Network (SGD)
    n_epochs = 30
    batch_size = 10
    gamma = 0
    eta = 0.00001
    lmbda = 0.0001

    nodes = np.arange(2, 20, 2)
    layers = np.arange(1, 3, 1)
    train_accuracy_score = np.zeros((len(nodes), len(layers)))
    test_accuracy_score = np.zeros((len(nodes), len(layers)))

    iter = 0
    for i, node in enumerate(nodes):
        for j, layer in enumerate(layers):

            # Need to create new instance, to change the architecture
            node_list = [node]*layer
            FFNN = Neural_Network(
                no_input_nodes=n_components,
                no_output_nodes=1,
                node_list=node_list
            )

            FFNN.set_cost_function(logistic_cost_NN)

            hidden_activation = 'Leaky_RELU'
            output_activation = 'sigmoid'
            FFNN.set_activation_function_hidden_layers(hidden_activation)
            FFNN.set_activation_function_output_layer(output_activation)

            # Changing some SGD values
            FFNN.set_SGD_values(
                n_epochs=n_epochs,
                batch_size=batch_size,
                gamma=gamma,
                eta=eta,
                lmbda=lmbda,
            )

            # Training the model
            FFNN.train_model(X_cancer_train, y_cancer_train)

            # Testing the model against the target values, and store the results
            FFNN.set_activation_function_output_layer(
                'sigmoid_classification')
            train_accuracy_score[i][j] = helper.accuracy_score(
                y_cancer_train, FFNN.feed_forward(X_cancer_train))
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_cancer_test, FFNN.feed_forward(X_cancer_test))

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(nodes) * len(layers)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_architecture(
        score=train_accuracy_score,
        x_tics=layers,
        y_tics=nodes,
        score_name='Training Accuracy',
        save_name=f'plots/test11/test11_M_{batch_size}_gamma_{gamma}_lmbda_{lmbda}_eta_{eta}_hidact_{hidden_activation}_outact_{output_activation}_training_7.png'
    )
    helper.seaborn_plot_architecture(
        score=test_accuracy_score,
        x_tics=layers,
        y_tics=nodes,
        score_name='Test Accuracy',
        save_name=f'plots/test11/test11_M_{batch_size}_gamma_{gamma}_lmbda_{lmbda}_eta_{eta}_hidact_{hidden_activation}_outact_{output_activation}_test_7.png'
    )


"""
TEST 12
DATASET: CANCER DATA (classification case)

Optimizing the batch_sizes and the momentum parameter gamma by looking over a seaborn plot. 
Will be measured in accuracy-score for both training and test-data. 
"""
test12 = False
if test12:
    print('>> RUNNING TEST 12:')
    # Loading the training and testing dataset
    n_components = 2
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = helper.load_cancer_data(
        n_components)

    # Setting up the Neural Network
    num_hidden_nodes = 20
    num_hidden_layers = 1
    node_list = [num_hidden_nodes]*num_hidden_layers
    FFNN = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=1,
        node_list=node_list
    )

    hidden_activation = 'sigmoid'
    FFNN.set_activation_function_hidden_layers(hidden_activation)

    # Set the parameters used in the Neural Network
    n_epochs = 300
    eta = 0.001
    lmbda = 0.0001
    FFNN.set_SGD_values(
        n_epochs=n_epochs,
        eta=eta,
        lmbda=lmbda
    )

    batch_sizes = np.arange(2, 32, 6)
    gammas = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    train_accuracy_score = np.zeros((len(batch_sizes), len(gammas)))
    test_accuracy_score = np.zeros((len(batch_sizes), len(gammas)))
    iter = 0

    for i, batch_size in enumerate(batch_sizes):
        for j, gamma in enumerate(gammas):
            # Refreshing the weights and biases before testing new SGD_values
            FFNN.initialize_the_biases()
            FFNN.initialize_the_weights()

            # Setting the preffered activation functions
            output_activation = 'sigmoid'
            FFNN.set_activation_function_output_layer(output_activation)

            # Set the preffered values of the gradient descent
            FFNN.set_SGD_values(
                batch_size=batch_size,
                gamma=gamma,
            )

            FFNN.train_model(X_cancer_train, y_cancer_train)

            # Testing the model against the target values, and store the results
            FFNN.set_activation_function_output_layer(
                'sigmoid_classification')

            # Finding the predicted values with our model
            y_hat_train = FFNN.feed_forward(X_cancer_train)
            y_hat_test = FFNN.feed_forward(X_cancer_test)

            train_accuracy_score[i][j] = helper.accuracy_score(
                y_cancer_train, y_hat_train)
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_cancer_test, y_hat_test)

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(batch_sizes) * len(gammas)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_batchsize_gamma(
        score=train_accuracy_score,
        x_tics=gammas,
        y_tics=batch_sizes,
        score_name='Training Accuracy',
        save_name=f'plots/test12/test12_M_{batch_size}_gamma_{gamma}_lmbda_{lmbda}_eta_{eta}_hidact_{hidden_activation}_outact_{output_activation}_training_3.png'
    )

    helper.seaborn_plot_batchsize_gamma(
        score=test_accuracy_score,
        x_tics=gammas,
        y_tics=batch_sizes,
        score_name='Test Accuracy',
        save_name=f'plots/test12/test12_M_{batch_size}_gamma_{gamma}_lmbda_{lmbda}_eta_{eta}_hidact_{hidden_activation}_outact_{output_activation}_test_3.png'
    )

"""
TEST 13
DATASET: FRANKE FUNCTION (regression case)

Analysis of the gradient descent algorithm, is scaling/ updating of
the learning rate necessary?
"""
test13 = False
if test13:
    print('>> RUNNING TEST 13:')
    n = 100
    noise = 0.01
    x_values, y_values, z_values = helper.generate_data(n, noise)

    n_epochs = 200
    degree = 5  # complexity of the model
    gamma = 0
    eta = 0.1
    batch_size = 10

    gradient_descent.main_OLS_scale_learning(
        x_values, y_values, z_values, n_epochs, degree, gamma, eta, batch_size)
