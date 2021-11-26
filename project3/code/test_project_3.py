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

import matplotlib.pyplot as plt
from FF_Neural_Network import Neural_Network
from gradient_descent import SGD
from cost_functions import logistic_cost_NN, logistic_cost_NN_multi, cost_logistic_regression, prob, \
    cost_logistic_regression_multi, prob_multi
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, BaggingRegressor
from sklearn.svm import SVC
from sklearn import tree
import time
import numpy as np
import gradient_descent
import helper
import seaborn as sns


"""
TEST 1:
DATASET: DIABETES DATA (classification case)
METHOD: Neural Network

Here you are able to try out the neural network and see the result in the form of
an accuracy score (of both training and test data). We have also provided the
opportunity to look at the accuracy score over the training time (iterations of the
SGD-algorithm)
"""
test1 = False
if test1:
    print('>> RUNNING TEST 1:')
    # Loading the training and testing dataset
    n_components = 2
    m_observations = 1000
    # X_train, X_test, y_train, y_test = helper.load_diabetes_data(
    #     n_components)
    
    X_train, X_test, y_train, y_test = helper.load_diabetes_data(
        n_components, m_observations)
    X_train, X_test, y_train, y_test = helper.load_cancer_data(
        n_components)

    # Setting the architecture of the Neural Network
    node_list = [3]*1

    # Initializing the Neural Network
    FFNN = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=2,
        node_list=node_list
    )

    # Setting the preffered Stochastic Gradient Descent parameters
    FFNN.set_SGD_values(
        n_epochs=300,
        batch_size=14,
        gamma=0.7,
        eta=1e-4,
        lmbda=1e-5)

    # Setting the preffered cost- and activation functions
    FFNN.set_cost_function(logistic_cost_NN_multi)
    FFNN.set_activation_function_hidden_layers('Leaky-RELU')
    FFNN.set_activation_function_output_layer('softmax')

    FFNN.train_model(X_train, y_train)
    y_hat = FFNN.feed_forward(X_train)
    
    y_pred_train = helper.convert_vec_to_num(FFNN.feed_forward(X_train))
    y_pred_test = helper.convert_vec_to_num(FFNN.feed_forward(X_test))

    for k in range(10):
        print(y_hat[k])
        print(y_train[k])
        print("---")

    print(helper.accuracy_score(y_train, y_pred_train))
    print(helper.accuracy_score(y_test, y_pred_test))

    # Change the activation function to predict 0 or 1's.
    # FFNN.set_activation_function_output_layer('sigmoid_classification')
    # print(
    #     f'Accuracy_train = {helper.accuracy_score(FFNN.feed_forward(X_train),  y_train)}')
    # print(
    #     f'Accuracy_test = {helper.accuracy_score(FFNN.feed_forward(X_test),  y_test)}')



"""
TEST 2:
DATASET: DIABETES DATA (classification case)
METHOD: Neural Network

Optimizing the architecture of the Neural Network (amount of hidden nodes and layers)
by looking over a seaborn plot. Will be measured in accuracy-score for both training and test-data.
"""
test2 = True
if test2:
    print('>> RUNNING TEST 2:')
    # Loading the training and testing dataset
    n_components = 5
    X_train, X_test, y_train, y_test = helper.load_diabetes_data(
        n_components, m_observations = 200)
    

    # X_train, X_test, y_train, y_test = helper.load_cancer_data(
    #     n_components)
    sns.set()

    # Set the parameters used in the Neural Network (SGD)
    n_epochs = 50
    batch_size = 10
    gamma = 0.6
    eta = 1e-3
    lmbda = 0.0001

    nodes = np.arange(2, 20, 2)
    layers = np.arange(1, 3, 1)
    train_accuracy_score = np.zeros((len(nodes), len(layers)))
    test_accuracy_score = np.zeros((len(nodes), len(layers)))

    iter = 0
    for i, node in enumerate(nodes):
        for j, layer in enumerate(layers):
            print(node, layer)
            # Need to create new instance, to change the architecture
            node_list = [node]*layer
            FFNN = Neural_Network(
                no_input_nodes=n_components,
                no_output_nodes=3,
                node_list=node_list
            )

            FFNN.set_cost_function(logistic_cost_NN_multi)

            hidden_activation = 'sigmoid'
            output_activation = 'softmax'
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
            FFNN.train_model(X_train, y_train)

            # Testing the model against the target values, and store the results
            y_pred_train = helper.convert_vec_to_num(FFNN.feed_forward(X_train))
            y_pred_test = helper.convert_vec_to_num(FFNN.feed_forward(X_test))
            
            train_accuracy_score[i][j] = helper.accuracy_score(
                y_train, y_pred_train)
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_test, y_pred_test)

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
TEST 3:
DATASET: DIABETES DATA (classification case)
METHOD: Neural Network

Optimizing the batch_sizes and the momentum parameter gamma by looking over a seaborn plot.
Will be measured in accuracy-score for both training and test-data.
"""
test3 = False
if test3:
    print('>> RUNNING TEST 3:')
    # Loading the training and testing dataset
    n_components = 2
    X_train, X_test, y_train, y_test = helper.load_cancer_data(
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

            FFNN.train_model(X_train, y_train)

            # Testing the model against the target values, and store the results
            FFNN.set_activation_function_output_layer(
                'sigmoid_classification')

            # Finding the predicted values with our model
            y_hat_train = FFNN.feed_forward(X_train)
            y_hat_test = FFNN.feed_forward(X_test)

            train_accuracy_score[i][j] = helper.accuracy_score(
                y_train, y_hat_train)
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_test, y_hat_test)

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
TEST 4:
DATASET: DIABETES DATA (classification case)
METHOD: Neural Network

Optimizing the hyperparameter lmbda and the learning rate by looking over
a seaborn plot. Will be measured in accuracy-score for both training and test-data.
"""

test4 = False
if test4:
    print('>> RUNNING TEST 4:')
    # Loading the training and testing dataset
    n_components = 2
    X_train, X_test, y_train, y_test = helper.load_cancer_data(
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
    max_depths = [0.0009, 0.0008, 0.0006, 0.0004, 0.0002, 0.0001, 0.00005]
    min_samples_leafs = np.logspace(-2, -7, 6)

    train_accuracy_score = np.zeros((len(max_depths), len(min_samples_leafs)))
    test_accuracy_score = np.zeros((len(max_depths), len(min_samples_leafs)))

    n_epochs = 200
    batch_size = 14
    gamma = 0.9

    FFNN.set_SGD_values(
        n_epochs=n_epochs,
        batch_size=batch_size,
        gamma=gamma)

    iter = 0
    for i, eta in enumerate(max_depths):
        for j, lmbda in enumerate(min_samples_leafs):
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
            FFNN.train_model(X_train, y_train)

            # Testing the model against the target values, and store the results
            FFNN.set_activation_function_output_layer(
                'sigmoid_classification')
            train_accuracy_score[i][j] = helper.accuracy_score(
                y_train, FFNN.feed_forward(X_train))
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_test, FFNN.feed_forward(X_test))

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(max_depths) * len(min_samples_leafs)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_lmbda_learning(
        score=train_accuracy_score,
        x_tics=min_samples_leafs,
        y_tics=max_depths,
        score_name='Training Accuracy',
        save_name=f'plots/test6/test6_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_features_{n_components}_hiddennodes_{no_hidden_nodes}_hiddenlayer_{no_hidden_layers}_actOUT_{output_activation}_training_14.png'
    )

    helper.seaborn_plot_lmbda_learning(
        score=test_accuracy_score,
        x_tics=min_samples_leafs,
        y_tics=max_depths,
        score_name='Test Accuracy',
        save_name=f'plots/test6/test6_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_features_{n_components}_hiddennodes_{no_hidden_nodes}_hiddenlayer_{no_hidden_layers}_actOUT_{output_activation}_test_14.png'
    )


"""
TEST 5:
DATASET: DIABETES DATA (classification case)
METHOD: Decision tree

Use scikitlearn's decision tree, testing
"""

test5 = False
if test5:
    print('>> RUNNING TEST 5:')
    # Loading the training and testing dataset
    n_components = 5
    m_observations = 300000
    X_train, X_test, y_train, y_test = helper.load_diabetes_data_without_PCA(
        n_components, m_observations)

    # Function to perform training with Entropy
    clf = DecisionTreeClassifier(
        criterion='entropy', max_depth=1)

    # Fit the data to the model we have created
    clf.fit(X_train, y_train)

    # Look at the three
    text_representation = tree.export_text(clf)
    print(text_representation)
    print(clf.tree_.max_depth)

    # Make predictions for
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(
        f'Accuracy_train = {helper.accuracy_score(y_pred_train, y_train)}')
    print(
        f'Accuracy_test = {helper.accuracy_score(y_pred_test,  y_test)}')

"""
TEST 6:
DATASET: DIABETES DATA (classification case)
METHOD: Decision tree

Training and test data vs. complexity of the tree
Use scikitlearn's decision tree, testing
"""

test6 = False
if test6:
    print('>> RUNNING TEST 6:')
    # Loading the training and testing dataset
    n_components = 2
    m_observations = 300000
    X_train, X_test, y_train, y_test = helper.load_diabetes_data_without_PCA(
        n_components, m_observations)

    max_depths = np.arange(1, 14, 1)

    train_accuracy_score = np.zeros(len(max_depths))
    test_accuracy_score = np.zeros(len(max_depths))

    iter = 0
    for i, max_depth in enumerate(max_depths):
        clf = DecisionTreeClassifier(
            criterion='entropy', max_depth=max_depth)

        # Fit the data to the model we have created
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        train_accuracy_score[i] = helper.accuracy_score(y_pred_train, y_train)
        test_accuracy_score[i] = helper.accuracy_score(y_pred_test, y_test)

        iter += 1
        print(
            f'Progress: {iter:2.0f}/{len(max_depths)}')

    plt.plot(max_depths,
             train_accuracy_score, label="Train accuracy score ")
    plt.plot(max_depths,
             test_accuracy_score, label="Test accuracy score ")

    plt.xlabel("Model Complexity (max depth of decision tree)")
    plt.ylabel("Accuracy score")
    plt.legend()
    plt.title(
        f"Accuracy vs max depth used in the decision tree algorithm")

    plt.show()


"""
TEST 7:
DATASET: DIABETES DATA (classification case)
METHOD: Random Forest

Use scikitlearn's decision tree, testing
"""

test7 = False
if test7:
    print('>> RUNNING TEST 7:')
    # Loading the training and testing dataset
    n_components = 5
    m_observations = 10000
    X_train, X_test, y_train, y_test = helper.load_diabetes_data(
        n_components, m_observations)

    # Create randomforest instance, with amount of max_depth
    clf = RandomForestClassifier(max_depth=2)

    # Fit the data to the model we have created
    clf.fit(X_train, y_train)

    # Make predictions for
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(
        f'Accuracy_train = {helper.accuracy_score(y_pred_train, y_train)}')
    print(
        f'Accuracy_test = {helper.accuracy_score(y_pred_test,  y_test)}')

"""
TEST 8:
DATASET: DIABETES DATA (classification case)
METHOD: Random forest

Training and test data vs. complexity of the tree
Use scikitlearn's decision tree
"""

test8 = False
if test8:
    print('>> RUNNING TEST 8:')
    # Loading the training and testing dataset
    n_components = 2
    m_observations = 300000
    X_train, X_test, y_train, y_test = helper.load_diabetes_data_without_PCA(
        n_components, m_observations)

    max_depths = np.arange(1, 40, 1)

    train_accuracy_score = np.zeros(len(max_depths))
    test_accuracy_score = np.zeros(len(max_depths))

    iter = 0
    for i, max_depth in enumerate(max_depths):
        clf = RandomForestClassifier(
            max_depth=max_depth)

        # Fit the data to the model we have created
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        train_accuracy_score[i] = helper.accuracy_score(y_pred_train, y_train)
        test_accuracy_score[i] = helper.accuracy_score(y_pred_test, y_test)

        iter += 1
        print(
            f'Progress: {iter:2.0f}/{len(max_depths)}')

    plt.plot(max_depths,
             train_accuracy_score, label="Train accuracy score ")
    plt.plot(max_depths,
             test_accuracy_score, label="Test accuracy score ")

    plt.xlabel("Model Complexity (max depth of random forest)")
    plt.ylabel("Accuracy score")
    plt.legend()
    plt.title(
        f"Accuracy vs max depth used in the decision tree algorithm")

    plt.show()


"""
TEST 9:
DATASET: DIABETES DATA (classification case)
METHOD: Gradient Boosting

Use scikitlearn's decision tree, testing
"""

test9 = False
if test9:
    print('>> RUNNING TEST 9:')
    # Loading the training and testing dataset
    n_components = 5
    m_observations = 10000
    X_train, X_test, y_train, y_test = helper.load_diabetes_data(
        n_components, m_observations)

    # Create randomforest instance, with amount of max_depth
    clf = GradientBoostingClassifier(
        n_estimators=10, learning_rate=0.1, max_depth=1)

    # Fit the data to the model we have created
    clf.fit(X_train, y_train)

    # Make predictions for
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(
        f'Accuracy_train = {helper.accuracy_score(y_pred_train, y_train)}')
    print(
        f'Accuracy_test = {helper.accuracy_score(y_pred_test,  y_test)}')

"""
TEST 10:
DATASET: DIABETES DATA (classification case)
METHOD: Bagging 

Training and test data vs. complexity of the tree
Use scikitlearn's decision tree
"""

test10 = False
if test10:
    print('>> RUNNING TEST 10:')
    # Loading the training and testing dataset
    n_components = 2
    m_observations = 300000
    X_train, X_test, y_train, y_test = helper.load_diabetes_data_without_PCA(
        n_components, m_observations)

    max_depths = np.arange(1, 21, 1)

    train_accuracy_score = np.zeros(len(max_depths))
    test_accuracy_score = np.zeros(len(max_depths))

    iter = 0
    for i, max_depth in enumerate(max_depths):
        clf = GradientBoostingClassifier(
            n_estimators=3, learning_rate=0.1, max_depth=max_depth, random_state=100)

        # Fit the data to the model we have created
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        train_accuracy_score[i] = helper.accuracy_score(y_pred_train, y_train)
        test_accuracy_score[i] = helper.accuracy_score(y_pred_test, y_test)

        iter += 1
        print(
            f'Progress: {iter:2.0f}/{len(max_depths)}')

    plt.plot(max_depths,
             train_accuracy_score, label="Train accuracy score ")
    plt.plot(max_depths,
             test_accuracy_score, label="Test accuracy score ")

    plt.xlabel("Model Complexity (max depth of random forest)")
    plt.ylabel("Accuracy score")
    plt.legend()
    plt.title(
        f"Accuracy vs max depth used in the decision tree algorithm")

    plt.show()

"""
TEST 11:
DATASET: DIABETES
METHOD: Logistic regression

Test if log reg works
"""
test11 = False
if test11:
    print(">> RUNNING TEST 11 <<")
    n_components = 4
    m_observations = 1000
    # X_train, X_test, y_train, y_test = helper.load_diabetes_data(n_components, m_observations)
    X_train, X_test, y_train, y_test = helper.load_diabetes_data(n_components, m_observations)
    n_classes = 3

    n_epochs = 400
    batch_size = 50
    gamma = 0.5
    iter = 0
    eta=1e-5
    lmbda=0

    theta, num = SGD(
        X=X_train, y=y_train,
        # theta_init=np.array([0.1]*(X_train.shape[1] + 1))
        theta_init = 0.01 + np.zeros((n_classes, X_train.shape[1] + 1)),
        eta=eta,
        cost_function=cost_logistic_regression_multi,
        n_epochs=n_epochs, batch_size=batch_size,
        gamma=gamma,
        lmbda=lmbda
    )

    y_pred_train = helper.convert_vec_to_num(prob_multi(theta, X_train))
    y_pred_test = helper.convert_vec_to_num(prob_multi(theta, X_test))

    print(f'Training accuracy: {helper.accuracy_score(y_train, y_pred_train)}')
    print(f'Testing accuracy: {helper.accuracy_score(y_test, y_pred_test)}')
    

    # for i in range(len(softi[:, 2])):
    #     if softi[i, ] > 0.3:
    #         print('-------')
    #         print(softi[i])
    #         print(y_train[i])
    #         print('--------')


