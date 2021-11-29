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

from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
from FF_Neural_Network import Neural_Network
from gradient_descent import SGD
from cost_functions import logistic_cost_NN, logistic_cost_NN_multi, cost_logistic_regression, prob, \
    cost_logistic_regression_multi, prob_multi, MSE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, BaggingRegressor
from sklearn.svm import SVC
from sklearn import tree
import time
import numpy as np
import gradient_descent
import helper
import seaborn as sns
# from tensorflow.keras.layers import Input
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras import optimizers
# from tensorflow.keras import regularizers
# from tensorflow.keras.utils import to_categorical


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
    n_components = 3
    X_train, X_test, y_train, y_test = helper.load_iris_data(n_components)

    # Setting the architecture of the Neural Network
    node_list = [12]

    # Initializing the Neural Network
    FFNN = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=3,
        node_list=node_list
    )

    # Setting the preffered Stochastic Gradient Descent parameters
    FFNN.set_SGD_values(
        n_epochs=2000,
        batch_size=5,
        gamma=0.1,
        eta=1e-2,
        lmbda=0)

    # Setting the preffered cost- and activation functions
    FFNN.set_cost_function(logistic_cost_NN_multi)
    FFNN.set_activation_function_hidden_layers('sigmoid')
    FFNN.set_activation_function_output_layer('softmax')

    FFNN.train_model(X_train, y_train, y_converter=True, keep_cost_values=True,
                     keep_accuracy_score=True)
    FFNN.plot_cost_of_last_training()
    FFNN.plot_accuracy_score_last_training()

    y_pred_train = helper.convert_vec_to_num(FFNN.feed_forward(X_train))
    y_pred_test = helper.convert_vec_to_num(FFNN.feed_forward(X_test))

    # for k in range(len(y_pred_test)):
    #     if y_train[k] == 0:
    #         print(
    #             f'predicted {y_pred_test[k][0]} : {int(y_test[k])} (exact). {y_pred_test[k][0]==int(y_test[k])}')
    #     if y_train[k] == 1:
    #         print(
    #             f'predicted {y_pred_test[k][0]} : {int(y_test[k])} (exact). {y_pred_test[k][0]==int(y_test[k])}')

    # print(y_pred_train)

    print(helper.accuracy_score(y_pred_train, y_train))
    print(helper.accuracy_score(y_test, y_pred_test))

    # Change the activation function to predict 0 or 1's.
    # FFNN.set_activation_function_output_layer('sigmoid_classification')


"""
TEST 2:
DATASET: DIABETES DATA (classification case)
METHOD: Neural Network

Optimizing the architecture of the Neural Network (amount of hidden nodes and layers)
by looking over a seaborn plot. Will be measured in accuracy-score for both training and test-data.
"""
test2 = False
if test2:
    print('>> RUNNING TEST 2:')
    # Loading the training and testing dataset
    n_components = 21
    X_train, X_test, y_train, y_test = helper.load_diabetes_data(
        n_components, m_observations=1000)

    # X_train, X_test, y_train, y_test = helper.load_cancer_data(
    #     n_components)
    sns.set()

    # Set the parameters used in the Neural Network (SGD)
    n_epochs = 300
    batch_size = 50
    gamma = 0.4
    eta = 4e-4
    lmbda = 0

    nodes = np.arange(2, 30, 2)
    layers = np.arange(1, 2, 1)
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
            y_pred_train = helper.convert_vec_to_num(
                FFNN.feed_forward(X_train))
            y_pred_test = helper.convert_vec_to_num(FFNN.feed_forward(X_test))

            train_accuracy_score[i][j] = helper.accuracy_score(
                y_train, y_pred_train)
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_test, y_pred_test)

            # if helper.accuracy_score(y_train, y_pred_train) < 0.37:
            #     for e, p, interval in zip(y_train, y_pred_train, FFNN.feed_forward(X_train)):
            #         print(f'(exact) {e} : {p} (predicted), {interval}')

            #     exit()
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
    # X_train, X_test, y_train, y_test = helper.load_cancer_data(
    #     n_components)

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
# TODO: change underneath
DATASET: DIABETES DATA (classification case)
METHOD: Neural Network

Optimizing the hyperparameter lmbda and the learning rate by looking over
a seaborn plot. Will be measured in accuracy-score for both training and test-data.
"""

test4 = False
if test4:
    print('>> RUNNING TEST 4:')
    # Loading the training and testing dataset
    n_components = 3
    X_train, X_test, y_train, y_test = helper.load_diabetes_data(
        n_components, 200)

    # Setting the architecture of the Neural Network
    no_hidden_nodes = 10
    no_hidden_layers = 1
    node_list = [no_hidden_nodes]*no_hidden_layers

    # Initializing the Neural Network
    FFNN = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=3,
        node_list=node_list
    )

    # Setting the preffered cost- and hidden_activation function
    FFNN.set_cost_function(logistic_cost_NN_multi)
    FFNN.set_activation_function_hidden_layers('sigmoid')

    # Change the activation function to predict 0 or 1's.
    max_depths = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
    lmbda_values = np.logspace(-5, -7, 3)

    train_accuracy_score = np.zeros((len(max_depths), len(lmbda_values)))
    test_accuracy_score = np.zeros((len(max_depths), len(lmbda_values)))

    n_epochs = 150
    batch_size = 10
    gamma = 0.1

    FFNN.set_SGD_values(
        n_epochs=n_epochs,
        batch_size=batch_size,
        gamma=gamma)

    iter = 0
    for i, eta in enumerate(max_depths):
        for j, lmbda in enumerate(lmbda_values):
            print(eta, lmbda)
            # Reinitializing the weights, biases and activation function
            output_activation = 'softmax'
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
            y_pred_test = helper.convert_vec_to_num(
                FFNN.feed_forward(X_train))
            y_pred_test = helper.convert_vec_to_num(FFNN.feed_forward(X_test))

            train_accuracy_score[i][j] = helper.accuracy_score(
                y_train, y_pred_test)
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_test, y_pred_test)

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(max_depths) * len(lmbda_values)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_lmbda_learning(
        score=train_accuracy_score,
        x_tics=lmbda_values,
        y_tics=max_depths,
        score_name='Training Accuracy',
        save_name=f'plots/test6/test6_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_features_{n_components}_hiddennodes_{no_hidden_nodes}_hiddenlayer_{no_hidden_layers}_actOUT_{output_activation}_training_14.png'
    )

    helper.seaborn_plot_lmbda_learning(
        score=test_accuracy_score,
        x_tics=lmbda_values,
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
    n_components = 3
    X_train, X_test, y_train, y_test = helper.load_iris_data(
        n_components, 2000)

    # Function to perform training with Entropy
    clf = DecisionTreeClassifier(
        criterion='entropy', max_depth=5)

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
    X_train, X_test, y_train, y_test = helper.load_iris_data(
        n_components)

    max_depths = np.arange(1, 14, 1)

    train_accuracy_score = []
    test_accuracy_score = []

    last_max_depth = 0
    iter = 0
    for i, max_depth in enumerate(max_depths):
        clf = DecisionTreeClassifier(
            criterion='entropy', max_depth=max_depth)

        # Fit the data to the model we have created
        clf.fit(X_train, y_train)
        max_depth = clf.tree_.max_depth

        if last_max_depth == max_depth:
            break
        else:
            last_max_depth = max_depth

        # Make predictions
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        train_accuracy_score.append(
            helper.accuracy_score(y_pred_train, y_train))
        test_accuracy_score.append(helper.accuracy_score(y_pred_test, y_test))

        iter += 1
        print(
            f'Progress: {iter:2.0f}/{len(max_depths)}')

    plt.plot(range(1, len(train_accuracy_score) + 1),
             train_accuracy_score, label="Train accuracy score ")
    plt.plot(range(1, len(test_accuracy_score) + 1),
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
    n_components = 3
    X_train, X_test, y_train, y_test = helper.load_iris_data(
        n_components)

    # Create randomforest instance, with amount of max_depth
    clf = RandomForestClassifier(max_depth=2)

    # Fit the data to the model we have created
    clf.fit(X_train, y_train)

    # Make predictions for
    y_pred_test = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(
        f'Accuracy_train = {helper.accuracy_score(y_pred_test, y_train)}')
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
    n_components = 3
    X_train, X_test, y_train, y_test = helper.load_iris_data(
        n_components)

    max_depths = np.arange(1, 40, 1)

    train_accuracy_score = []
    test_accuracy_score = []

    iter = 0
    last_train_accuracy = 0
    for i, max_depth in enumerate(max_depths):
        clf = RandomForestClassifier(
            max_depth=max_depth)

        # Fit the data to the model we have created
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        train_accuracy = helper.accuracy_score(y_pred_train, y_train)
        if last_train_accuracy == 1:
            break
        else:
            last_train_accuracy = train_accuracy

        train_accuracy_score.append(train_accuracy)
        test_accuracy_score.append(helper.accuracy_score(y_pred_test, y_test))

        iter += 1
        print(
            f'Progress: {iter:2.0f}/{len(max_depths)}')

    plt.plot(range(1, len(train_accuracy_score) + 1),
             train_accuracy_score, label="Train accuracy score ")
    plt.plot(range(1, len(test_accuracy_score) + 1),
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
    n_components = 3
    X_train, X_test, y_train, y_test = helper.load_iris_data(n_components)

    n_classes = 3

    n_epochs = 3000
    batch_size = 10
    gamma = 0.8
    iter = 0
    eta = 1e-3
    lmbda = 0

    theta, num = SGD(
        X=X_train, y=y_train,
        # theta_init=np.array([0.1]*(X_train.shape[1] + 1))
        theta_init=0.01 + np.zeros((n_classes, X_train.shape[1] + 1)),
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


# """
# TEST 12:
# DATASET:Bean
# METHOD: NN Keras

# Tensorflow Keras
# """
# test12 = True
# if test12:
#     print(">> RUNNING TEST 11 <<")
#     # Loading the training and testing dataset
#     n_components = 2
#     m_observations = 1600

#     X_train, X_test, y_train, y_test = helper.load_dry_beans_data(2)
#     y_train = to_categorical(y_train)
#     y_test = to_categorical(y_test)

#     model = Sequential()
#     model.add(Dense(20, activation="relu", input_dim=n_components))
#     # model.add(Dense(20, activation="relu"))
#     # model.add(Dense(64, activation="relu"))
#     model.add(Dense(7, activation='softmax'))
#     sgd = optimizers.SGD(learning_rate=1e-1, momentum=0.7)

#     model.compile(loss='categorical_crossentropy',
#                   optimizer=sgd,
#                   metrics=['accuracy'])

#     model.fit(X_train, y_train,
#               epochs=500,
#               batch_size=100)

#     train_scores = model.evaluate(X_train, y_train, batch_size=100)
#     test_scores = model.evaluate(X_test, y_test, batch_size=100)
#     print(f"\nAccuracy for training: {train_scores[1]}")
#     print(f"Accuracy for testing: {test_scores[1]}")

"""
TEST 13:
Dataset: Iris
Method: Logistic regression

Grid search gamma batch size
"""

test13 = False
if test13:
    n_components = 3
    X_train, X_test, y_train, y_test = helper.load_iris_data(n_components)

    n_classes = 3

    n_epochs = 100
    eta = 1e-3
    lmbda = 0

    gamma_values = [round(0.2*i, 1) for i in range(5)]
    batch_sizes = [2, 5, 10, 20]

    train_accuracy_score = np.zeros((len(batch_sizes), len(gamma_values)))
    test_accuracy_score = np.zeros((len(batch_sizes), len(gamma_values)))

    iter = 0
    for i, batch_size in enumerate(batch_sizes):
        for j, gamma in enumerate(gamma_values):
            theta, num = SGD(
                X=X_train, y=y_train,
                theta_init=0.01 + np.zeros((n_classes, X_train.shape[1] + 1)),
                eta=eta,
                cost_function=cost_logistic_regression_multi,
                n_epochs=n_epochs, batch_size=batch_size,
                gamma=gamma,
                lmbda=lmbda
            )

            y_pred_train = helper.convert_vec_to_num(
                prob_multi(theta, X_train))
            y_pred_test = helper.convert_vec_to_num(
                prob_multi(theta, X_test))

            train_accuracy_score[i][j] = helper.accuracy_score(
                y_train, y_pred_train)
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_test, y_pred_test)

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(batch_sizes) * len(gamma_values)}')

    helper.seaborn_plot_batchsize_gamma(
        score=train_accuracy_score,
        x_tics=gamma_values,
        y_tics=batch_sizes,
        score_name='Training Accuracy',
        save_name=f'plots/test13/test13_{lmbda}_eta_{eta}_training.png'
    )

    helper.seaborn_plot_batchsize_gamma(
        score=test_accuracy_score,
        x_tics=gamma_values,
        y_tics=batch_sizes,
        score_name='Test Accuracy',
        save_name=f'plots/test13/test13_gamma_{gamma}_lmbda_{lmbda}_test_.png'
    )


"""
TEST 14:
DATASET: IRIS (classification case)
METHOD: Logistic regression

Optimizing the hyperparameter lmbda and the learning rate by looking over
a seaborn plot. Will be measured in accuracy-score for both training and test-data.
"""

test14 = False
if test14:
    print('>> RUNNING TEST 14:')
    # Loading the training and testing dataset
    n_components = 3
    X_train, X_test, y_train, y_test = helper.load_iris_data(n_components)

    n_classes = 3

    # Change the activation function to predict 0 or 1's.
    learning_rates = [10**(-i) for i in range(5)]
    lmbda_values = np.logspace(-1, -6, 6)

    train_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))
    test_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))

    n_epochs = 100
    batch_size = 10
    gamma = 0.8

    iter = 0
    for i, eta in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbda_values):
            theta, num = SGD(
                X=X_train, y=y_train,
                theta_init=0.01 + np.zeros((n_classes, X_train.shape[1] + 1)),
                eta=eta,
                cost_function=cost_logistic_regression_multi,
                n_epochs=n_epochs, batch_size=batch_size,
                gamma=gamma,
                lmbda=lmbda
            )

            y_pred_train = helper.convert_vec_to_num(
                prob_multi(theta, X_train))
            y_pred_test = helper.convert_vec_to_num(
                prob_multi(theta, X_test))

            train_accuracy_score[i][j] = helper.accuracy_score(
                y_train, y_pred_train)
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_test, y_pred_test)

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(learning_rates) * len(lmbda_values)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_lmbda_learning(
        score=train_accuracy_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Training Accuracy',
        save_name=f'plots/test14/test14_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_features_{n_components}_training_14.png'
    )

    helper.seaborn_plot_lmbda_learning(
        score=test_accuracy_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Test Accuracy',
        save_name=f'plots/test14/test14_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_features_{n_components}_test_14.png'
    )


"""
TEST 15:
DATASET: IRIS (classification case)

Show the explained variance ratio from the IRIS dataset 
with PCA
"""
test15 = False
if test15:
    print('>> RUNNING TEST 14:')

    n_components_list = [1, 2, 3, 4]
    for n_components in n_components_list:
        helper.load_iris_data(n_components, show_explained_ratio=True)


"""
TEST 16:
DATASET: Housing Data
METHOD: Neural Network

Testing NN for housing data
"""

test16 = True
if test16:
    print('>> RUNNING TEST 16:')
    # Loading the training and testing dataset
    n_components = 2
    X_train, X_test, y_train, y_test = helper.load_housing_california_data(
        n_components, 100)

    # Setting the architecture of the Neural Network
    node_list = [5]

    # Initializing the Neural Network
    FFNN = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=1,
        node_list=node_list
    )

    # Setting the preffered Stochastic Gradient Descent parameters
    FFNN.set_SGD_values(
        n_epochs=1000,
        batch_size=5,
        gamma=0.4,
        eta=1e-3,
        lmbda=0)

    # Setting the preffered cost- and activation functions
    FFNN.set_cost_function(MSE)
    FFNN.set_activation_function_hidden_layers('sigmoid')

    FFNN.train_model(X_train, y_train, keep_cost_values=True)
    FFNN.plot_cost_of_last_training()

    y_pred_train = FFNN.feed_forward(X_train)
    y_pred_test = FFNN.feed_forward(X_test)

    for i, j in zip(y_pred_train, y_train):
        print(f'(pred) {i} - {j} (exact). DIFF = {abs(i-j)}')

    print(helper.r2_score(y_pred_train, y_train))
    print(helper.r2_score(y_test, y_pred_test))

"""
TEST 17:
DATASET: IRIS (regression case)
METHOD: 

Plotting the MSE, bias and variance (testing data), 
over complexity
"""
test17 = False
if test17:
    print('>> RUNNING TEST 17:')
    # Spliting in training and testing data
    X_train, X_test, z_train, z_test = helper.load_housing_data(2, 5000)
    x_train = X_train[:, 0]
    y_train = X_train[:, 1]
    x_test = X_test[:, 0]
    y_test = X_test[:, 1]

    # Store the MSE, bias and variance values for test data
    list_of_MSE_testing = []
    list_of_bias_testing = []
    list_of_variance_testing = []

    max_degree = 4
    n_bootstrap = 100
    lmbda = 0.1

    # Finding MSE, bias and variance of test data for different degrees
    for degree in range(1, max_degree + 1):

        # Create matrix for storing the bootstrap results
        z_pred_test_matrix = np.empty((z_test.shape[0], n_bootstrap))

        # Running bootstrap-method
        for i in range(n_bootstrap):

            _x, _y, _z = helper.resample(x_train, y_train, z_train)

            # Predicting z with the training set, _x, _y, _z
            z_pred_test, _, _ = helper.predict_output(
                x_train=_x, y_train=_y, z_train=_z,
                x_test=x_test, y_test=y_test,
                degree=degree, regression_method='OLS',
                lmbda=lmbda
            )

            z_pred_test_matrix[:, i] = z_pred_test

        # Finding MSE, bias and variance from the bootstrap
        MSE_test = np.mean(np.mean(
            (z_pred_test_matrix - np.transpose(np.array([z_test])))**2))

        bias_test = np.mean(
            (z_test - np.mean(z_pred_test_matrix, axis=1, keepdims=False))**2)

        variance_test = np.mean(
            np.var(z_pred_test_matrix, axis=1, keepdims=False))

        list_of_MSE_testing.append(MSE_test)
        list_of_bias_testing.append(bias_test)
        list_of_variance_testing.append(variance_test)

    plt.plot(range(1, max_degree + 1),
             list_of_MSE_testing, label="Test sample - error")
    plt.plot(range(1, max_degree + 1),
             list_of_bias_testing, label="Test sample - bias")
    plt.plot(range(1, max_degree + 1),
             list_of_variance_testing, label="Test sample - variance")
    plt.xlabel("Model Complexity")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.title(
        f"bias-variance trade-off vs model complexity\n\
        Data points: {len(x_train)}; Method: {'OLS'}"
    )

    plt.show()
    plt.close()
