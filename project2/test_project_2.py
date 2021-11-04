"""
Setup for testing the results of project 2, feel free to change the
different variables. Read docstrings inside the functions
to know their functionality
"""

import time
import numpy as np
import gradient_descent
import helper
import seaborn as sns
import matplotlib.pyplot as plt
from FF_Neural_Network import Neural_Network


"""
TEST 1
Analysis of the gradien descent algorithm, and comparing it
against the results of project 1. 
"""
test1 = False
if test1:
    # Generating some data (Franke Function)
    n = 500
    noise = 0.4
    x_values, y_values, z_values = helper.generate_data(n, noise)

    # Setting some preffered values
    list_number_of_minibatches = [1, 10, 20, 40,
                                  100, 400]
    list_number_of_minibatches = [1, 400]
    number_of_epochs = 20
    degree = 1  # complexity of the model
    gamma = 0.1  # the momentum of the stochastic gradient decent

    "Set to true, stochastic gradient decent testing with OLS"
    run_main_OLS = False
    if run_main_OLS:
        gradient_descent.main_OLS(
            x_values=x_values, y_values=y_values, z_values=z_values,
            list_no_of_minibatches=list_number_of_minibatches,
            n_epochs=number_of_epochs,
            degree=degree, gamma=gamma
        )

    no_of_minibatches = 10
    "Set to true, stochastic gradient decent testing with RIDGE"
    run_main_RIDGE = False
    if run_main_RIDGE:
        gradient_descent.main_RIDGE(
            x_values=x_values, y_values=y_values, z_values=z_values,
            no_of_minibatches=no_of_minibatches,
            n_epochs=number_of_epochs,
            degree=degree, gamma=gamma
        )


"""
TEST 2
Analysis of the Neural Network class, with data from the
Franke Function in project 1
"""
test2 = False
if test2:
    # Initializing some data
    n = 200
    noise = 0
    x1, x2, y = helper.generate_data(n, noise)
    X = np.array(list(zip(x1, x2)))
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    # Initializing the Neural Network
    number_of_input_nodes = 2
    number_of_output_nodes = 1
    number_of_hidden_nodes = 5
    number_of_hidden_layers = 2
    FFNN = Neural_Network(
        no_input_nodes=number_of_input_nodes,
        no_output_nodes=number_of_output_nodes,
        no_hidden_nodes=number_of_hidden_nodes,
        no_hidden_layers=number_of_hidden_layers,
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
    keep_cost_values = True
    FFNN.train_model(X_train, y_train, keep_cost_values)

    if keep_cost_values:
        # If True: we can plot the cost vs. iterations
        FFNN.plot_cost_of_last_training()

    # Finding the predicted values with our model
    y_hat_train = FFNN.feed_forward(X_train)
    y_hat_test = FFNN.feed_forward(X_test)

    # Checking the result
    print('\n\n>>> CHECKING THE MODEL: \n')
    print('>> (R2) TRAINING DATA: ', end='')
    print(helper.r2_score(y_train, y_hat_train))
    print(f'>> (R2) TESTING DATA: ', end='')
    print(helper.r2_score(y_hat_test, y_test))


"""
TEST 3
START: the testing of Neural Network vs. RIDGE and OLS
"""
test3 = False
if test3:
    # Initializing some data
    n = 200
    noise = 0
    x1, x2, y = helper.generate_data(n, noise)
    X = np.array(list(zip(x1, x2)))
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    # Initializing the Neural Network
    neural_start = time.time()
    number_of_input_nodes = 2
    number_of_output_nodes = 1
    number_of_hidden_nodes = 10
    number_of_hidden_layers = 2
    FFNN = Neural_Network(
        no_input_nodes=number_of_input_nodes,
        no_output_nodes=number_of_output_nodes,
        no_hidden_nodes=number_of_hidden_nodes,
        no_hidden_layers=number_of_hidden_layers,
    )

    # Setting the preffered activation functions (for hidden and output layer)
    FFNN.set_activation_function_hidden_layers('sigmoid')

    # Set the preffered values of the gradient descent
    FFNN.set_SGD_values(
        eta=5e-4,
        lmbda=1e-6,
        n_epochs=4000,
        batch_size=8,
        gamma=0.6
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
Testing the regression case, analysis of the learning rate and 
the regularization parameter
"""
test4 = True
if test4:
    # Initializing some data
    n = 200
    noise = 0
    x1, x2, y = helper.generate_data(n, noise)
    X = np.array(list(zip(x1, x2)))
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = helper.train_test_split(X, y)

    # Initializing the Neural Network
    number_of_input_nodes = 2
    number_of_output_nodes = 1
    number_of_hidden_nodes = 5
    number_of_hidden_layers = 2
    FFNN = Neural_Network(
        no_input_nodes=number_of_input_nodes,
        no_output_nodes=number_of_output_nodes,
        no_hidden_nodes=number_of_hidden_nodes,
        no_hidden_layers=number_of_hidden_layers,
    )

    # Setting the preffered activation functions for hidden layer
    FFNN.set_activation_function_hidden_layers('sigmoid')

    # Set the preffered values of the gradient descent
    FFNN.set_SGD_values(
        n_epochs=400,
        batch_size=10,
        gamma=0.7
    )
    # Starting up a seaborn plot
    sns.set()
    # TODO: set those underneath optimal -> many times there will be predicting the same values
    # SO PROBABLY CANCEL THE EXIT()
    learning_rates = np.logspace(-2, -5, 4)
    lmbda_values = np.logspace(-3, -6, 4)

    train_R2_score = np.zeros((len(learning_rates), len(lmbda_values)))
    test_R2_score = np.zeros((len(learning_rates), len(lmbda_values)))
    for i, eta in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbda_values):

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

    fig, ax = plt.subplots(figsize=(8, 8))
    heat = sns.heatmap(train_R2_score, annot=True, ax=ax, cmap="viridis",
                       xticklabels=lmbda_values, yticklabels=learning_rates)

    # The some
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    # plt.savefig("project2/plots/RIDGE_heatmap_training_2.png")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(test_R2_score, annot=True, ax=ax, cmap="viridis",
                xticklabels=lmbda_values, yticklabels=learning_rates)
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    # plt.savefig("project2/plots/RIDGE_heatmap_testing_2.png")
    plt.show()


"""
TEST 5:
# TODO: task c
Classification network
Learning rate vs lmbda seaborn plot 
can set different activiation functions, and NEW DATA Set
"""

"""
TEST 6:
logistisc and classification network different in time 

"""

"""
TEST 7: 
Logistic regression, seaborn plot, 
learning rate and lmbda

"""
