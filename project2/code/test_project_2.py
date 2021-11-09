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
from logistic_regression import cost_logistic_regression, prob
from gradient_descent import SGD
from cost_functions import logistic_cost
import time
import numpy as np
import gradient_descent
import helper
import seaborn as sns


"""
TEST 1
Analysis of the gradient descent algorithm, and comparing it
against the results of project 1. For the Ridge case, it will provide
a seaborn visualization with hyperparameter lambda and the learning rate
"""
test1 = False
if test1:
    print('>> RUNNING TEST 1:')
    # Generating some data (Franke Function)
    n = 500
    noise = 0.4
    x_values, y_values, z_values = helper.generate_data(n, noise)
    number_of_epochs = 20
    degree = 1  # complexity of the model
    gamma = 0.1  # the momentum of the stochastic gradient decent

    "Set to true, stochastic gradient decent testing with OLS"
    run_main_OLS = False
    if run_main_OLS:
        print('> Analysing the gradient descent algorithm with OLS')

        # Set the number of minibatches you want to analyse
        list_number_of_minibatches = [1, n*0.8]
        gradient_descent.main_OLS(
            x_values=x_values, y_values=y_values, z_values=z_values,
            list_no_of_minibatches=list_number_of_minibatches,
            n_epochs=number_of_epochs,
            degree=degree, gamma=gamma
        )

    no_of_minibatches = 10
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
Analysis of the Neural Network class, with data from the
Franke Function in project 1. Here you can set the parameters as
preffered. You will be provided with the plot of the value of cost-function
vs. the iteration number (of SGD) in the training of the model (if some keep-boolean is true). 
You will also get the R2-score of the training and testing data.
"""
test2 = False
if test2:
    print('>> RUNNING TEST 2:')
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
    print('\n\n>>> CHECKING THE MODEL:')
    print('>> (R2) TRAINING DATA: ', end='')
    print(helper.r2_score(y_train, y_hat_train))
    print(f'>> (R2) TESTING DATA: ', end='')
    print(helper.r2_score(y_hat_test, y_test))


"""
TEST 3
Testing the Neural Network, vs. OLS- and RIDGE-regression. It will 
provide the results from the three different methods in R2-score. And the 
time spent for creating these models. 
"""
test3 = False
if test3:
    print('>> RUNNING TEST 3:')
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
Testing the regression case, analysis of the hyperparameter lambda
and the learning rate. The result is shown in a seaborn plot. You get
both the R2-score for training- and test-data. 
"""
test4 = False
if test4:
    print('>> RUNNING TEST 4:')
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
        n_epochs=40,
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

    # Creating the seaborn_plot
    helper.seaborn_plot(
        score=train_R2_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name="Training R2-score",
        # save_name='Ridge_heatmap_training_5.png'
    )
    helper.seaborn_plot(
        score=test_R2_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name="Test R2-score",
        # save_name='Ridge_heatmap_training_5.png'
    )


"""
TEST 5:
Classification Problem: here are you able to set different activation functions, 
cost-functions, and different parameters for the Neural Network and SGD. It will, 
also provide the accuracy_score of the training and testing data - and here we consider 
the breast-cancer data from scikit learn. A plot will be provided, if not turned of 
(set keep_accuracy = False). This will show the accuracy of the training vs. amount of
iterations in the SGD.
"""

test5 = False
if test5:
    print('>> RUNNING TEST 5:')
    # Loading the training and testing dataset
    n_components = 2
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = helper.load_cancer_data(
        n_components)

    # Setting the architecture of the Neural Network
    no_hidden_nodes = 2
    no_hidden_layers = 1

    # Initializing the Neural Network
    FFNN = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=1,
        no_hidden_nodes=no_hidden_nodes,
        no_hidden_layers=no_hidden_layers
    )

    # Setting the preffered Stochastic Gradient Descent parameters
    FFNN.set_SGD_values(
        n_epochs=20,
        batch_size=10,
        gamma=0.8,
        eta=0.01,
        lmbda=1e-5)

    # Setting the preffered cost- and activation functions
    FFNN.set_cost_function(logistic_cost)
    # FFNN.set_activation_function_hidden_layers('sigmoid')
    FFNN.set_activation_function_output_layer('sigmoid')

    # Set keep_accuracy to True, if you wanna see accuracy vs. time (in the training)
    keep_accuracy = True
    FFNN.train_model(X_cancer_train, y_cancer_train,
                     keep_accuracy_score=keep_accuracy)

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
Classification problem: the same as in the last TESTING, but with a seaborn plot. The plot
will show the accuracy with some combination of learning rates and the regularization 
parameter lambda. All the other parameters are to be selected as preffered. 
"""

test6 = False
if test6:
    print('>> RUNNING TEST 6:')
    # Loading the training and testing dataset
    n_components = 2
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = helper.load_cancer_data(
        n_components)

    # Setting the architecture of the Neural Network
    no_hidden_nodes = 2
    no_hidden_layers = 1

    # Initializing the Neural Network
    FFNN = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=1,
        no_hidden_nodes=no_hidden_nodes,
        no_hidden_layers=no_hidden_layers
    )

    # Setting the preffered Stochastic Gradient Descent parameters
    FFNN.set_SGD_values(
        n_epochs=20,
        batch_size=10,
        gamma=0.8,
        eta=0.01,
        lmbda=1e-5)

    # Setting the preffered cost- and hidden_activation function
    FFNN.set_cost_function(logistic_cost)
    # FFNN.set_activation_function_hidden_layers('sigmoid')

    # Change the activation function to predict 0 or 1's.
    learning_rates = np.logspace(-3, -5, 3)
    lmbda_values = np.logspace(-3, -5, 3)

    train_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))
    test_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))

    FFNN.set_SGD_values(
        n_epochs=10,
        batch_size=10,
        gamma=0.8)
    for i, eta in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbda_values):
            # Reinitializing the weights, biases and activation function
            FFNN.set_activation_function_output_layer('sigmoid')
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

    # Creating the seaborn_plot
    helper.seaborn_plot(
        score=train_accuracy_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Training Accuracy',
        # save_name='BREAST_heatmap_training_5.png'
    )
    helper.seaborn_plot(
        score=test_accuracy_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Test Accuracy',
        # save_name='BREAST_heatmap_testing_5.png'
    )


"""
TEST 7:
Logistic regression: create a seaborn plot of the hyperparameter lambda
and the learning rate. Other SGD values, will be easy to change in the test. 
"""
test7 = False
if test7:
    # Loading the training and testing dataset
    n_components = 2
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = helper.load_cancer_data(
        n_components)

    # TODO: this shall change, and shall be able to have more features
    X_train_design = helper.create_design_matrix(
        X_cancer_train[:, 0], X_cancer_train[:, 1], degree=1)

    X_test_design = helper.create_design_matrix(
        X_cancer_test[:, 0], X_cancer_test[:, 1], degree=1)

    learning_rates = np.logspace(-3, -5, 3)
    lmbda_values = np.logspace(-3, -5, 3)

    train_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))
    test_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))

    for i, eta in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbda_values):

            theta, num = SGD(
                X=X_train_design, y=y_cancer_train,
                theta_init=np.array([0.1]*X_train_design.shape[1]),
                eta=eta,
                cost_function=cost_logistic_regression,
                n_epochs=30, M=10,
                gamma=0.8,
                lmbda=lmbda
            )

            # Finding the predicted values
            predicted_values_train = np.where(
                prob(theta, X_train_design) >= 0.5, 1, 0)
            predicted_values_test = np.where(
                prob(theta, X_test_design) >= 0.5, 1, 0)

            # Applying the model against the target values, and store the results
            train_accuracy_score[i][j] = helper.accuracy_score(
                y_cancer_train, predicted_values_train)
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_cancer_test, predicted_values_test)

    # Creating the seaborn_plot
    helper.seaborn_plot(
        score=train_accuracy_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Training Accuracy',
        save_name='seaborn_plot_logistic_breast_training_1.png'
    )
    helper.seaborn_plot(
        score=test_accuracy_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Test Accuracy',
        save_name='seaborn_plot_logistic_breast_test_1.png'
    )


"""
TEST 8: 
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
    no_hidden_nodes = 2
    no_hidden_layers = 1

    # Initializing the Neural Network
    FFNN = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=1,
        no_hidden_nodes=no_hidden_nodes,
        no_hidden_layers=no_hidden_layers
    )

    # Setting the preffered Stochastic Gradient Descent parameters
    FFNN.set_SGD_values(
        n_epochs=20,
        batch_size=10,
        gamma=0.8,
        eta=0.01,
        lmbda=1e-5)

    # Setting the preffered cost- and activation functions
    FFNN.set_cost_function(logistic_cost)
    # FFNN.set_activation_function_hidden_layers('sigmoid')
    FFNN.set_activation_function_output_layer('sigmoid')
    FFNN.train_model(X_cancer_train, y_cancer_train)

    # Change the activation function to predict 0 or 1's.
    FFNN.set_activation_function_output_layer('sigmoid_classification')
    print('>>>>> Results of classification problem with Neural Network <<<<<')
    print(
        f'ACCURACY_train = {helper.accuracy_score(FFNN.feed_forward(X_cancer_train),  y_cancer_train): .4f}')
    print(
        f'ACCURACY_test = {helper.accuracy_score(FFNN.feed_forward(X_cancer_test),  y_cancer_test): .4f}')
    print(f'TIME SPENT: {time.time() - time_NN_start : .3f}')

    "Secondly, look at the logistic regression and find the results"
    time_logistic_start = time.time()
    # TODO: this shall change, and shall be able to have more features
    X_train_design = helper.create_design_matrix(
        X_cancer_train[:, 0], X_cancer_train[:, 1], degree=1)

    X_test_design = helper.create_design_matrix(
        X_cancer_test[:, 0], X_cancer_test[:, 1], degree=1)

    theta, num = SGD(
        X=X_train_design, y=y_cancer_train,
        theta_init=np.array([0.1]*X_train_design.shape[1]),
        eta=0.01,
        cost_function=cost_logistic_regression,
        n_epochs=30, M=10,
        gamma=0.8,
        lmbda=1e-4
    )

    # Finding the predicted values
    predicted_values_train = np.where(prob(theta, X_train_design) >= 0.5, 1, 0)
    predicted_values_test = np.where(prob(theta, X_test_design) >= 0.5, 1, 0)

    print('\n>>>>> Results of logistic regression, with SGD <<<<<')
    print(
        f'ACCURACY_train => {helper.accuracy_score(predicted_values_train, y_cancer_train): .4f}')
    print(
        f'ACCURACY_test => {helper.accuracy_score(predicted_values_test, y_cancer_test): .4f}')
    print(f'TIME SPENT: {time.time() - time_logistic_start: .3f}')
