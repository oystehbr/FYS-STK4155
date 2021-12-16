"""
Setup for testing the results of project 3, feel free to change the
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

import time
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from gradient_descent import SGD
from cost_functions import cost_logistic_regression_multi, prob_multi
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import tree
import numpy as np
import helper
import seaborn as sns
import pandas as pd

"""
TEST 1:
DATASET: Beans data (classification case)
METHOD: Neural Network

Here you are able to try out the neural network and see the result in the form of
an accuracy score (of both training and test data). In addition, we have provided 
the opportunity to look at the confusion matrix of the testing data
"""
test1 = False
if test1:
    print('>> RUNNING TEST 1:')
    # Loading the training and testing dataset
    n_components = 3
    m_observations = 20000
    X_train, X_test, y_train, y_test = helper.load_dry_beans_data(
        n_components, m_observations)

    # Switching the target values to a vector (other representation of the output)
    y_train = to_categorical(y_train, num_classes=7)
    y_test = to_categorical(y_test, num_classes=7)

    n_epochs = 30
    batch_size = 100
    lmbda = 1e-4
    eta = 1e-2
    gamma = 0.9
    hidden_nodes = 14
    hidden_layers = 4
    model = helper.create_NN(n_components, hidden_nodes, hidden_layers,
                             'categorical_crossentropy', 'relu', eta, lmbda, gamma)

    # Fitting the model
    model.fit(X_train, y_train,
              epochs=n_epochs,
              batch_size=batch_size)

    y_hat_train = helper.convert_vec_to_num(model.predict(X_train))
    y_hat_test = helper.convert_vec_to_num(model.predict(X_test))
    y_train = helper.convert_vec_to_num(y_train)
    y_test = helper.convert_vec_to_num(y_test)

    train_accuracy_score = helper.accuracy_score(
        y_train, y_hat_train)
    test_accuracy_score = helper.accuracy_score(
        y_test, y_hat_test)
    print(f"\nAccuracy for training: {train_accuracy_score}")
    print(f"Accuracy for testing: {test_accuracy_score}")

    # Provide a confusion matrix if True
    confusion_result = True
    if confusion_result:
        array = confusion_matrix(y_test, y_hat_test)

        df_cm = pd.DataFrame(array, range(7), range(7))
        plt.figure(figsize=(8, 8))
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(df_cm, annot=True, annot_kws={
                    "size": 12}, fmt='d', cmap='Blues')  # font size
        plt.title(f'Confusion matrix for beans data (Neural Network)')
        plt.xlabel("classes")
        plt.ylabel("classes")

        plt.savefig('plots/test1/test1_confusion_matrix_NN_optimal.png')
        plt.show()


"""
TEST 2:
DATASET: Beans data (classification case)
METHOD: Neural Network

Optimizing the architecture of the Neural Network (amount of hidden nodes and layers)
by looking over a seaborn plot. Will be measured in accuracy-score for both training and test-data.
"""
test2 = False
if test2:
    print('>> RUNNING TEST 2:')
    # Loading the training and testing dataset
    n_components = 3
    m_observations = 13611
    X_train, X_test, y_train, y_test = helper.load_dry_beans_data(
        n_components, m_observations)

    sns.set()

    # Set the parameters used in the Neural Network (SGD)
    n_epochs = 80
    batch_size = 500
    gamma = 0.8
    eta = 1e-2
    lmbda = 1e-4
    act_func = 'relu'

    nodes = np.arange(2, 20, 4)
    layers = np.arange(2, 14, 2)
    train_accuracy_score = np.zeros((len(nodes), len(layers)))
    test_accuracy_score = np.zeros((len(nodes), len(layers)))

    iter = 0
    for i, node in enumerate(nodes):
        for j, layer in enumerate(layers):
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
            # Need to create new instance, to change the architecture
            FFNN = helper.create_NN(n_components, node, layer,
                                    'categorical_crossentropy', act_func, eta, lmbda, gamma)

            # Training the model
            FFNN.fit(X_train, y_train, epochs=n_epochs,
                     batch_size=batch_size, verbose=0)

            y_hat_train = helper.convert_vec_to_num(FFNN.predict(X_train))
            y_hat_test = helper.convert_vec_to_num(FFNN.predict(X_test))
            y_train = helper.convert_vec_to_num(y_train)
            y_test = helper.convert_vec_to_num(y_test)

            train_accuracy_score[i][j] = helper.accuracy_score(
                y_train, y_hat_train)
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_test, y_hat_test)

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(nodes) * len(layers)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_architecture(
        score=train_accuracy_score,
        x_tics=layers,
        y_tics=nodes,
        score_name='Training Accuracy',
        save_name=f'plots/test2/test2_M_{batch_size}_gamma_{gamma}_lmbda_{lmbda}_eta_{eta}_epochs_{n_epochs}_training_8.png'
    )
    helper.seaborn_plot_architecture(
        score=test_accuracy_score,
        x_tics=layers,
        y_tics=nodes,
        score_name='Test Accuracy',
        save_name=f'plots/test2/test2_M_{batch_size}_gamma_{gamma}_lmbda_{lmbda}_eta_{eta}_epochs_{n_epochs}_test_8.png'
    )


"""
TEST 3:
DATASET: Beans data (classification case)
METHOD: Neural Network

Optimizing the batch_sizes and the momentum parameter gamma by looking over a seaborn plot.
Will be measured in accuracy-score for both training and test-data.
"""
test3 = False
if test3:
    print('>> RUNNING TEST 3:')
    # Loading the training and testing dataset
    n_components = 3
    m_observations = 13611
    X_train, X_test, y_train, y_test = helper.load_dry_beans_data(
        n_components, m_observations)

    sns.set()

    # Initialize some values
    eta = 1e-2
    lmbda = 1e-4
    act_func = 'relu'
    hidden_nodes = 14
    hidden_layers = 4

    batch_sizes = np.arange(100, 1100, 200)
    gammas = [0, 0.2, 0.6, 0.8, 0.9, 1.0]
    train_accuracy_score = np.zeros((len(batch_sizes), len(gammas)))
    test_accuracy_score = np.zeros((len(batch_sizes), len(gammas)))

    iter = 0
    for i, batch_size in enumerate(batch_sizes):
        for j, gamma in enumerate(gammas):

            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)

            # Need to create new instance, to change the architecture
            FFNN = helper.create_NN(n_components, hidden_nodes, hidden_layers,
                                    'categorical_crossentropy', act_func, eta, lmbda, gamma)

            # Training the model
            FFNN.fit(X_train, y_train, epochs=int(batch_size/20),
                     batch_size=batch_size, verbose=0)

            # Predicting
            y_hat_train = helper.convert_vec_to_num(FFNN.predict(X_train))
            y_hat_test = helper.convert_vec_to_num(FFNN.predict(X_test))
            y_train = helper.convert_vec_to_num(y_train)
            y_test = helper.convert_vec_to_num(y_test)

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
        save_name=f'plots/test3/test3_lmbda_{lmbda}_eta_{eta}_training_1.png'
    )

    helper.seaborn_plot_batchsize_gamma(
        score=test_accuracy_score,
        x_tics=gammas,
        y_tics=batch_sizes,
        score_name='Test Accuracy',
        save_name=f'plots/test3/test3_lmbda_{lmbda}_eta_{eta}_test_1.png'
    )


"""
TEST 4:
DATASET: Beans data (classification case)
METHOD: Neural Network

Optimizing the hyperparameter lmbda and the learning rate by looking over
a seaborn plot. Will be measured in accuracy-score for both training and test-data.
"""
test4 = False
if test4:
    print('>> RUNNING TEST 4:')
    # Loading the training and testing dataset
    n_components = 3
    m_observations = 13611
    X_train, X_test, y_train, y_test = helper.load_dry_beans_data(
        n_components, m_observations)

    sns.set()

    # Change the activation function to predict 0 or 1's.
    learning_rates = np.logspace(0, -5, 6)
    lmbda_values = np.logspace(-2, -7, 6)

    train_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))
    test_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))

    # Initialize some values
    n_epochs = 20
    gamma = 0.9
    batch_size = 100
    act_func = 'leaky_relu'
    hidden_nodes = 14
    hidden_layers = 4

    iter = 0
    for i, eta in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbda_values):

            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)

            # Need to create new instance, to change the architecture
            FFNN = helper.create_NN(n_components, hidden_nodes, hidden_layers,
                                    'categorical_crossentropy', act_func, eta, lmbda, gamma)

            # Training the model
            FFNN.fit(X_train, y_train, epochs=n_epochs,
                     batch_size=batch_size, verbose=0)

            # Predicting
            y_hat_train = helper.convert_vec_to_num(FFNN.predict(X_train))
            y_hat_test = helper.convert_vec_to_num(FFNN.predict(X_test))
            y_train = helper.convert_vec_to_num(y_train)
            y_test = helper.convert_vec_to_num(y_test)

            train_accuracy_score[i][j] = helper.accuracy_score(
                y_train, y_hat_train)
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_test, y_hat_test)

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(learning_rates) * len(lmbda_values)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_lmbda_learning(
        score=train_accuracy_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Training Accuracy',
        save_name=f'plots/test4/test4nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_training.png'
    )

    helper.seaborn_plot_lmbda_learning(
        score=test_accuracy_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Test Accuracy',
        save_name=f'plots/test4/test4_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_test.png'
    )


"""
TEST 5:
DATASET: Beans data (classification case)
METHOD: Decision tree

Print tree representation, the accuracy-score and the depth used and 
the test will also provide a confusion matrix if interested. 
"""
test5 = False
if test5:
    print('>> RUNNING TEST 5:')
    # Loading the training and testing dataset
    n_components = 3
    m_observations = 13611
    X_train, X_test, y_train, y_test = helper.load_dry_beans_data(
        n_components, m_observations)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # Function to perform training with Entropy
    max_depth = 5
    clf = DecisionTreeClassifier(
        criterion='entropy', max_depth=max_depth)

    # Fit the data to the model we have created
    clf.fit(X_train, y_train)

    # Look at the tree
    look_at_tree = False
    if look_at_tree:
        text_representation = tree.export_text(clf)
        print(text_representation)
        print(clf.tree_.max_depth)

    # Make predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(
        f'Accuracy_train = {helper.accuracy_score(y_pred_train, y_train)}')
    print(
        f'Accuracy_test = {helper.accuracy_score(y_pred_test,  y_test)}')

    confusion_result = True
    if confusion_result:
        array = confusion_matrix(y_test, y_pred_test)

        df_cm = pd.DataFrame(array, range(7), range(7))
        plt.figure(figsize=(8, 8))
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(df_cm, annot=True, annot_kws={
                    "size": 12}, fmt='d', cmap='Blues')  # font size
        plt.title(f'Confusion matrix for beans data (Decision tree)')
        plt.xlabel("classes")
        plt.ylabel("classes")

        plt.savefig(
            'plots/test5/test5_confusion_matrix_decision_tree_optimal.png')
        plt.show()

"""
TEST 6:
DATASET: Beans data (classification case)
METHOD: Decision tree

Optimizing the max_depth parameter - looking on the accuracy
against the complexity parameter: 'max_depth'. The max_depth will be lower
than the number provided if the decision tree is accurate on all targets (on training data). 
Therefore, we stop the iterations, when the model is all accurate on the training data
"""
test6 = False
if test6:
    print('>> RUNNING TEST 6:')
    # Loading the training and testing dataset
    n_components = 3
    m_observations = 13611
    X_train, X_test, y_train, y_test = helper.load_dry_beans_data(
        n_components)

    max_depths = np.arange(1, 100, 1)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

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

    plt.savefig('plots/test6/test6_accuracy_vs_decision_tree_classification_2')
    plt.show()


"""
TEST 7:
DATASET: Beans data (classification case)
METHOD: Random Forest

Print the accuracy-score and the depth used, and the test
will also provide a confusion matrix.
"""

test7 = False
if test7:
    print('>> RUNNING TEST 7:')
    # Loading the training and testing dataset
    n_components = 3
    m_observations = 13611
    X_train, X_test, y_train, y_test = helper.load_dry_beans_data(
        n_components, m_observations)

    # Converting the type to integer
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # Create randomforest instance, with amount of max_depth
    max_depth = 6
    clf = RandomForestClassifier(max_depth=max_depth)

    # Fit the data to the model we have created
    clf.fit(X_train, y_train)

    # Make predictions for
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(
        f'Accuracy_train = {helper.accuracy_score(y_pred_train, y_train)}')
    print(
        f'Accuracy_test = {helper.accuracy_score(y_pred_test,  y_test)}')

    # Setting up the confusion matrix for testing data
    confusion_result = True
    if confusion_result:
        array = confusion_matrix(y_test, y_pred_test)

        df_cm = pd.DataFrame(array, range(7), range(7))
        plt.figure(figsize=(8, 8))
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(df_cm, annot=True, annot_kws={
                    "size": 12}, fmt='d', cmap='Blues')  # font size
        plt.title(f'Confusion matrix for beans data (Decision tree)')
        plt.xlabel("classes")
        plt.ylabel("classes")

        plt.savefig(
            'plots/test7/test7_confusion_matrix_random_forest_optimal_1.png')
        plt.show()

"""
TEST 8:
DATASET: Beans data (classification case)
METHOD: Random forest

Optimizing the max_depth parameter - looking on the accuracy
against the complexity parameter: 'max_depth'. Here we are using 
the default amount of trees (from the packages), which is 
'n_estimators = 100' (by default).
"""
test8 = False
if test8:
    print('>> RUNNING TEST 8:')

    # Loading the training and testing dataset
    n_components = 3
    m_observations = 13611
    X_train, X_test, y_train, y_test = helper.load_dry_beans_data(
        n_components)

    # Converting the types to integer
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # Setting up the values for the iterations and the storage
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

        # Check if we have gotten max accuracy on testing data (then stop)
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
    plt.savefig('plots/test8/test8_random_forest_accuracy_vs_complexity.png')

    plt.show()

"""
TEST 9:
DATASET: Beans data (classification case)

Test for showing both the distribution of the data (the amount of targets
of the different classes), and it will also provide the 'explained variance
ratio' for the amount of components provided in the list:
n_components_list
This is test is for finding the optimal number of components to use in
the machine learning algorithm. 
"""
test9 = False
if test9:
    print('>> RUNNING TEST 9:')
    helper.load_dry_beans_data(2, show_target_distribution=True)

    # Setting the number of components to check the total explained variance ratio of.
    n_components_list = [1, 2, 3]
    for n_components in n_components_list:
        helper.load_dry_beans_data(n_components, show_explained_ratio=True)

"""
TEST 10:
DATASET: Beans data (classification case)

Test the time different of the four methods:
- neural network 
- logistic regression
- decision tree
- random forest

"""
test10 = False
if test10:
    print('>> RUNNING TEST 10:')
    # Loading the training and testing dataset
    n_components = 3
    m_observations = 13611
    X_train, X_test, y_train, y_test = helper.load_dry_beans_data(
        n_components, m_observations)

    neural_network = True
    time_neural_start = time.time()
    if neural_network:
        # Switching the target values to a vector (other representation of the output)
        y_train_NN = to_categorical(y_train, num_classes=7)

        n_epochs = 30
        batch_size = 100
        lmbda = 1e-5
        eta = 1e-2
        gamma = 0.9
        hidden_nodes = 14
        hidden_layers = 4
        model = helper.create_NN(n_components, hidden_nodes, hidden_layers,
                                 'categorical_crossentropy', 'relu', eta, lmbda, gamma)

        # Fitting the model
        model.fit(X_train, y_train_NN,
                  epochs=n_epochs,
                  batch_size=batch_size)
    time_neural = time.time() - time_neural_start
    print('>> Time: {time_neural} s (Neural network)')

    logistic_regression = True
    time_logistic_start = time.time()
    if logistic_regression:
        y_train_reg = y_train.astype('int')

        n_classes = 7

        # Initialize some variables
        n_epochs = 200
        batch_size = 500
        gamma = 0.8
        eta = 1e-2
        lmbda = 1e-5
        iter = 0

        theta, num = SGD(
            X=X_train, y=y_train_reg,
            # theta_init=np.array([0.1]*(X_train.shape[1] + 1))
            theta_init=0.01 + np.zeros((n_classes, X_train.shape[1] + 1)),
            eta=eta,
            cost_function=cost_logistic_regression_multi,
            n_epochs=n_epochs, batch_size=batch_size,
            gamma=gamma,
            lmbda=lmbda
        )
    time_logistic = time.time() - time_logistic_start
    print('>> Time: {time_logistic} s (Logistic regression)')

    decision_tree = True
    time_tree_start = time.time()
    if decision_tree:
        y_train_decision_tree = y_train.astype('int')

        # Function to perform training with Entropy
        max_depth = 5
        clf = DecisionTreeClassifier(
            criterion='entropy', max_depth=max_depth)

        # Fit the data to the model we have created
        clf.fit(X_train, y_train_decision_tree)
    time_tree = time.time() - time_tree_start
    print('>> Time: {time_tree} s (Decision tree)')

    random_forest = True
    time_forest_start = time.time()
    if random_forest:
        # Converting the type to integer
        y_train_random_forest = y_train.astype('int')

        # Create randomforest instance, with amount of max_depth
        max_depth = 6
        clf = RandomForestClassifier(max_depth=max_depth)

        # Fit the data to the model we have created
        clf.fit(X_train, y_train_random_forest)
    time_forest = time.time() - time_forest_start
    print('>> Time: {time_forest} s (Random forest)')


"""
TEST 11:
DATASET: Beans data (classification case)
METHOD: Logistic regression

Logistic regression, look at the accuracy score with the parameters 
provide inside the test below. You can also be provided with a confusion 
matrix of the testing data. 
"""

test11 = False
if test11:
    print(">> RUNNING TEST 11 <<")
    # Loading the data
    n_components = 3
    m_observations = 13611
    X_train, X_test, y_train, y_test = helper.load_dry_beans_data(
        n_components, m_observations)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    n_classes = 7

    # Initialize some variables
    n_epochs = 200
    batch_size = 500
    gamma = 0.8
    eta = 1e-2
    lmbda = 1e-5
    iter = 0

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

    # Finding the prediction of the model and printing the result
    y_pred_train = helper.convert_vec_to_num(prob_multi(theta, X_train))
    y_pred_test = helper.convert_vec_to_num(prob_multi(theta, X_test))

    print(f'Training accuracy: {helper.accuracy_score(y_train, y_pred_train)}')
    print(f'Testing accuracy: {helper.accuracy_score(y_test, y_pred_test)}')

    # If True, then the confusion matrix will be computed and shown
    confusion_result = True
    if confusion_result:
        array = confusion_matrix(y_test, y_pred_test)

        df_cm = pd.DataFrame(array, range(7), range(7))
        plt.figure(figsize=(8, 8))
        sns.set(font_scale=1.4)  # for label size
        sns.heatmap(df_cm, annot=True, annot_kws={
                    "size": 12}, fmt='d', cmap='Blues')  # font size
        plt.title(f'Confusion matrix for beans data (Logistic regression)')
        plt.xlabel("classes")
        plt.ylabel("classes")

        plt.savefig('plots/test11/tes11_confusion_matrix_logistic_optimal.png')
        plt.show()

"""
TEST 12:
Dataset: Beans data (classification case)
Method: Logistic regression

Grid search of the momentum parameter gamma, and the batch size
used in the SGD algorithm for optimizing the model. 
"""

test12 = False
if test12:
    print(">> RUNNING TEST 12 <<")
    # Loading the data
    n_components = 3
    m_observations = 13611
    X_train, X_test, y_train, y_test = helper.load_dry_beans_data(
        n_components, m_observations)

    # Setting up the structure
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    n_classes = 7

    # Initialize some values
    eta = 1e-3
    lmbda = 0

    gamma_values = [round(0.2*i, 1) for i in range(5)]
    batch_sizes = np.arange(100, 1100, 200)

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
                n_epochs=batch_size, batch_size=batch_size,
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
        save_name=f'plots/test12/test12_{lmbda}_eta_{eta}_training.png'
    )

    helper.seaborn_plot_batchsize_gamma(
        score=test_accuracy_score,
        x_tics=gamma_values,
        y_tics=batch_sizes,
        score_name='Test Accuracy',
        save_name=f'plots/test12/test12_gamma_{gamma}_lmbda_{lmbda}_test_.png'
    )


"""
TEST 13:
DATASET: Beans data (classification case)
METHOD: Logistic regression

Optimizing the hyperparameter lmbda and the learning rate by looking over
a seaborn plot. Will be measured in accuracy-score for both training and test-data.
"""

test13 = False
if test13:
    print('>> RUNNING TEST 13:')
    # Loading the training and testing dataset
    n_components = 3
    m_observations = 13611
    X_train, X_test, y_train, y_test = helper.load_dry_beans_data(
        n_components, m_observations)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    n_classes = 7

    # Change the activation function to predict 0 or 1's.
    learning_rates = [10**(-i) for i in range(1, 5)]
    lmbda_values = np.logspace(-1, -6, 6)

    train_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))
    test_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))

    # Initialize some variables
    n_epochs = 100
    batch_size = 500
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
        save_name=f'plots/test13/test13_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_features_{n_components}_training_14.png'
    )

    helper.seaborn_plot_lmbda_learning(
        score=test_accuracy_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Test Accuracy',
        save_name=f'plots/test13/test13_nepochs_{n_epochs}_M_{batch_size}_gamma_{gamma}_features_{n_components}_test_14.png'
    )


"""
TEST 14:
DATASET: Housing data (regression case)
METHOD: Neural network

Using neural network by tensorflow, to predict prices. Can check different
kind of networks and optimization parameters. It will provide the R2-score
for both the testing- and training data.
"""
test14 = False
if test14:
    print(">> RUNNING TEST 14 <<")
    # Loading the training and testing dataset
    n_components = 8
    m_observations = 2000

    X_train, X_test, y_train, y_test = helper.load_housing_california_data(
        n_components, m_observations)

    lmbda = 1e-4
    eta = 5e-4
    gamma = 0.8
    hidden_nodes = 60
    hidden_layers = 7
    model = helper.create_NN(n_components, hidden_nodes, hidden_layers, 'mse',
                             "relu", eta, lmbda, gamma)

    model.fit(X_train, y_train,
              epochs=500,
              batch_size=100, )

    y_hat_train = model.predict(X_train)
    y_hat_test = model.predict(X_test)

    r2_score_train = helper.r2_score(y_train, y_hat_train)
    r2_score_test = helper.r2_score(y_test, y_hat_test)

    print(f"\nR2-score training: {r2_score_train}")
    print(f"R2-score testing: {r2_score_test}")

"""
TEST 16:
DATASET: Housing data (regression case)
METHOD: Decision tree

Print tree representation, the r2-score and the depth used
in the tree
"""

test16 = False
if test16:
    print('>> RUNNING TEST 16:')
    # Loading the training and testing dataset
    n_components = 4
    X_train, X_test, y_train, y_test = helper.load_housing_california_data(
        n_components, m_observations=2000, show_explained_ratio=True)

    # Function to perform training with Entropy
    clf = DecisionTreeRegressor(max_depth=100)

    # Fit the data to the model we have created
    clf.fit(X_train, y_train)

    # Look at the three
    text_representation = tree.export_text(clf)
    print(text_representation)
    print(clf.tree_.max_depth)

    # Make predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(helper.r2_score(y_pred_train, y_train))
    print(helper.r2_score(y_test, y_pred_test))


"""
TEST 17:
DATASET: Housing data (regression case)
METHOD: Decision tree

Optimizing the algorithm by looking over the
depth of the tree.
"""

test17 = False
if test17:
    print('>> RUNNING TEST 17:')
    # Loading the training and testing dataset
    n_components = 8
    X_train, X_test, y_train, y_test = helper.load_housing_california_data(
        n_components, 2000)

    max_depths = np.arange(1, 30, 1)

    train_accuracy_score = []
    test_accuracy_score = []

    last_max_depth = 0
    iter = 0
    for i, max_depth in enumerate(max_depths):
        clf = DecisionTreeRegressor(
            max_depth=max_depth, criterion="mse")

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
            helper.r2_score(y_pred_train, y_train))
        test_accuracy_score.append(helper.r2_score(y_pred_test, y_test))

        iter += 1
        print(
            f'Progress: {iter:2.0f}/{len(max_depths)}')

    plt.plot(range(1, len(train_accuracy_score) + 1),
             train_accuracy_score, label="Train R2-score ")
    plt.plot(range(1, len(test_accuracy_score) + 1),
             test_accuracy_score, label="Test R2-score ")

    plt.xlabel("Model Complexity (max depth of decision tree)")
    plt.ylabel("R2-score")
    plt.legend()
    plt.title(
        f"Accuracy vs max depth used in the decision tree algorithm")

    plt.show()

"""
TEST 18:
DATASET: Housing data (regression case)
METHOD: Random Forest

Evaluating the random forest model, set the depth to the
preffered value and see the result (in the R2-score)
"""

test18 = False
if test18:
    print('>> RUNNING TEST 18:')
    # Loading the training and testing dataset
    n_components = 8
    X_train, X_test, y_train, y_test = helper.load_housing_california_data(
        n_components, 2000)

    # Create randomforest instance, with amount of max_depth
    clf = RandomForestRegressor(max_depth=2)

    # Fit the data to the model we have created
    clf.fit(X_train, y_train)

    # Make predictions for
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(
        f'R2_score_train = {helper.r2_score(y_pred_train, y_train)}')
    print(
        f'R2_score_test = {helper.r2_score(y_pred_test,  y_test)}')


"""
TEST 19:
DATASET: Housing data (regression case)
METHOD: Random forest

Optimizing the random forest, by looking over the
depth of the trees. The amount of trees are set by default 
to 100. 
"""

test19 = False
if test19:
    print('>> RUNNING TEST 19:')
    # Loading the training and testing dataset
    n_components = 8
    X_train, X_test, y_train, y_test = helper.load_housing_california_data(
        n_components, 1000)

    max_depths = np.arange(4, 100, 2)

    train_accuracy_score = []
    test_accuracy_score = []

    iter = 0
    last_train_accuracy = 0
    for i, max_depth in enumerate(max_depths):
        clf = RandomForestRegressor(
            max_depth=max_depth, random_state=50)

        # Fit the data to the model we have created
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        train_accuracy = helper.r2_score(y_pred_train, y_train)
        if last_train_accuracy == 1:
            break
        else:
            last_train_accuracy = train_accuracy

        train_accuracy_score.append(train_accuracy)
        test_accuracy_score.append(helper.r2_score(y_pred_test, y_test))

        iter += 1
        print(
            f'Progress: {iter:2.0f}/{len(max_depths)}')

    plt.plot(range(1, len(train_accuracy_score) + 1),
             train_accuracy_score, label="Train R2-score ")
    plt.plot(range(1, len(test_accuracy_score) + 1),
             test_accuracy_score, label="Test R2-score ")

    plt.xlabel("Model Complexity (max depth of random forest)")
    plt.ylabel("R2 score")
    plt.legend()
    plt.title(
        f"R2_score vs max depth used in the decision tree algorithm")

    plt.show()

"""
TEST 20:
DATASET: Housing data (regression case)
METHOD: OLS regression

Bias-variance tradeoff (not used)
"""

test20 = False
if test20:
    print('>> RUNNING TEST 20:')
    n_components = 2
    m_observations = 10000
    X_train, X_test, z_train, z_test = helper.load_housing_california_data(
        n_components, m_observations)
    x_train = X_train[:, 0]
    y_train = X_train[:, 1]
    x_test = X_test[:, 0]
    y_test = X_test[:, 1]

    # Store the MSE, bias and variance values for test data
    list_of_MSE_testing = []
    list_of_bias_testing = []
    list_of_variance_testing = []

    max_degree = 5
    n_bootstrap = 100
    lmbda = 0.001
    method = 'OLS'
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
                degree=degree, regression_method=method,
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
        Data points: {len(x_train) + len(x_test)}; Method: {method}"
    )
    plt.savefig('plots/test20/test20_OLS_bias_variance_1.png')
    plt.show()
    plt.close()


"""
TEST 21:
DATASET: Housing data (regression case)
METHOD: Neural Network

Bias-variance tradeoff. The error vs. the complexity of the method,
where the complexity in the neural network is the amount of hidden layers
"""

test21 = False
if test21:
    print('>> RUNNING TEST 21:')
    n_components = 8
    m_observations = 2000
    X_train, X_test, y_train, y_test = helper.load_housing_california_data(
        n_components, m_observations)

    # Store the MSE, bias and variance values for test data
    list_of_MSE_testing = []
    list_of_bias_testing = []
    list_of_variance_testing = []

    max_degree = 12
    n_bootstraps = 10
    iter = 0

    lmbda = 1e-4
    eta = 5e-4
    gamma = 0.85
    hidden_nodes = 60
    batch_size = 120
    epochs = 500
    # Finding MSE, bias and variance of test data for different degrees
    for degree in range(2, max_degree + 1, 2):

        # Create matrix for storing the bootstrap results
        y_pred_test_matrix = np.empty((y_test.shape[0], n_bootstraps))

        # Running bootstrap-method
        for i in range(n_bootstraps):

            _X, _y, = helper.resample(X_train, y_train)

            # create NN model using tensorflow/keras
            model = helper.create_NN(
                n_components, hidden_nodes, degree, 'mse', "relu", eta, lmbda, gamma)
            model.fit(_X, _y,
                      epochs=epochs,
                      batch_size=batch_size,)

            y_pred_test_matrix[:, i] = model.predict(X_test).reshape(1, -1)
            iter += 1
            print(
                f"---------------- {iter}/{max_degree * n_bootstraps} RUNS ----------------")

        # Finding MSE, bias and variance from the bootstrap
        MSE_test = np.mean(np.mean(
            (y_pred_test_matrix - np.transpose(np.array([y_test])))**2))

        bias_test = np.mean(
            (y_test - np.mean(y_pred_test_matrix, axis=1, keepdims=False))**2)

        variance_test = np.mean(
            np.var(y_pred_test_matrix, axis=1, keepdims=False))

        list_of_MSE_testing.append(MSE_test)
        list_of_bias_testing.append(bias_test)
        list_of_variance_testing.append(variance_test)

    plt.plot(range(2, max_degree + 1, 2),
             list_of_MSE_testing, label="Test sample - error")
    plt.plot(range(2, max_degree + 1, 2),
             list_of_bias_testing, label="Test sample - bias")
    plt.plot(range(2, max_degree + 1, 2),
             list_of_variance_testing, label="Test sample - variance")
    plt.xlabel("Model Complexity")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.title(
        f"bias-variance trade-off vs model complexity\n\
        Data points: {len(X_train[:,0]) + len(X_test[:,0])}; Method: Neural Network"
    )
    plt.savefig(
        f"plots/test21/bias_variance_boots_NN.png")
    plt.show()
    plt.close()


"""
TEST 22:
DATASET: Housing data (regression case)
METHOD: Neural Network

Optimizing the batch_sizes and the momentum parameter gamma by looking over a seaborn plot.
Will be measured in r2-score for both training and test-data.
"""
test22 = False
if test22:
    print('>> RUNNING TEST 22:')
    # Loading the training and testing dataset
    n_components = 8
    m_observations = 2000
    X_train, X_test, y_train, y_test = helper.load_housing_california_data(
        n_components, m_observations)

    batch_sizes = np.arange(40, 440, 40)
    gammas = [0, 0.4, 0.6, 0.8, 0.9]

    train_r2_score = np.zeros((len(batch_sizes), len(gammas)))
    test_r2_score = np.zeros((len(batch_sizes), len(gammas)))

    # Initialize some values
    eta = 5e-4
    lmbda = 1e-5
    hidden_nodes = 60
    hidden_layers = 7
    act_func = 'relu'

    iter = 0
    for i, batch_size in enumerate(batch_sizes):
        for j, gamma in enumerate(gammas):
            # Creating a neural network model
            model = helper.create_NN(
                n_components, hidden_nodes, hidden_layers, 'mse', act_func, eta, lmbda, gamma)

            epochs = batch_size
            # Train the model
            model.fit(X_train, y_train,
                      epochs=batch_size*2,
                      batch_size=batch_size,)

            y_hat_train = model.predict(X_train)
            y_hat_test = model.predict(X_test)

            train_r2_score[i][j] = helper.r2_score(
                y_train, y_hat_train)
            test_r2_score[i][j] = helper.r2_score(
                y_test, y_hat_test)

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(batch_sizes) * len(gammas)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_batchsize_gamma(
        score=train_r2_score,
        x_tics=gammas,
        y_tics=batch_sizes,
        score_name='Training R2-score',
        save_name=f'plots/test22/test22_M_{batch_size}_gamma_{gamma}_lmbda_{lmbda}_eta_{eta}_training_3.png'
    )

    helper.seaborn_plot_batchsize_gamma(
        score=test_r2_score,
        x_tics=gammas,
        y_tics=batch_sizes,
        score_name='Test R2-score',
        save_name=f'plots/test22/test22_M_{batch_size}_gamma_{gamma}_lmbda_{lmbda}_eta_{eta}_test_3.png'
    )

"""
TEST 23:
DATASET: Housing data (regression case)
METHOD: Neural Network

Optimizing the hyperparameter lmbda and the learning rate by looking over
a seaborn plot. Will be measured in r2-score for both training and test-data.
"""

test23 = False
if test23:
    print('>> RUNNING TEST 23:')
    # Loading the training and testing dataset
    m_observations = 2000
    n_components = 8
    X_train, X_test, y_train, y_test = helper.load_housing_california_data(
        n_components, m_observations)

    learning_rates = [10**(-i) for i in range(1, 5)]
    lmbda_values = np.logspace(-1, -7, 7)

    train_r2_score = np.zeros((len(learning_rates), len(lmbda_values)))
    test_r2_score = np.zeros((len(learning_rates), len(lmbda_values)))

    # Initialize some variables
    batch_size = 120
    epochs = 500
    gamma = 0.8
    hidden_nodes = 60
    hidden_layers = 7
    act_func = 'relu'

    iter = 0
    for i, eta in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbda_values):
            # Creating the neural network model
            model = helper.create_NN(
                n_components, hidden_nodes, hidden_layers, 'mse', act_func, eta, lmbda, gamma)

            # Train the model
            model.fit(X_train, y_train,
                      epochs=epochs,
                      batch_size=batch_size,)

            y_hat_train = model.predict(X_train)
            y_hat_test = model.predict(X_test)

            train_r2_score[i][j] = helper.r2_score(
                y_train, y_hat_train)
            test_r2_score[i][j] = helper.r2_score(
                y_test, y_hat_test)

            iter += 1
            print(
                f'Progress: {iter:2.0f}/{len(learning_rates) * len(lmbda_values)}')

    # Creating the seaborn_plot
    helper.seaborn_plot_lmbda_learning(
        score=train_r2_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Training R2-score',
        save_name=f'plots/test23/test23_nepochs_{epochs}_M_{batch_size}_gamma_{gamma}_features_{n_components}_training_14.png'
    )

    helper.seaborn_plot_lmbda_learning(
        score=test_r2_score,
        x_tics=lmbda_values,
        y_tics=learning_rates,
        score_name='Test R2-score',
        save_name=f'plots/test23/test23_nepochs_{epochs}_M_{batch_size}_gamma_{gamma}_features_{n_components}_test_14.png'
    )


"""
TEST 24:
DATASET: Housing data (regression case)
METHOD: Decision tree

Bias-variance tradeoff. The error vs. the complexity of the method,
where the complexity in decision tree is the depth of the trees. 
"""

test24 = False
if test24:
    print('>> RUNNING TEST 24:')
    n_components = 8
    m_observations = 2000
    X_train, X_test, y_train, y_test = helper.load_housing_california_data(
        n_components, m_observations)

    # Store the MSE, bias and variance values for test data
    list_of_MSE_testing = []
    list_of_bias_testing = []
    list_of_variance_testing = []

    iter = 0
    max_degree = 20
    n_bootstraps = 1000
    # Finding MSE, bias and variance of test data for different degrees
    for degree in range(1, max_degree + 1):

        # Create matrix for storing the bootstrap results
        y_pred_test_matrix = np.empty((y_test.shape[0], n_bootstraps))

        # Running bootstrap-method
        for i in range(n_bootstraps):

            _X, _y, = helper.resample(X_train, y_train)

            # Function to perform training with decision tree
            clf = DecisionTreeRegressor(max_depth=degree)

            # Fit the data to the model we have created
            clf.fit(_X, _y)

            # Make predictions
            y_pred_test_matrix[:, i] = clf.predict(X_test).reshape(1, -1)

        iter += 1
        print(f"---------------- {iter}/{max_degree} RUNS ----------------")

        # Finding MSE, bias and variance from the bootstrap
        MSE_test = np.mean(np.mean(
            (y_pred_test_matrix - np.transpose(np.array([y_test])))**2))

        bias_test = np.mean(
            (y_test - np.mean(y_pred_test_matrix, axis=1, keepdims=False))**2)

        variance_test = np.mean(
            np.var(y_pred_test_matrix, axis=1, keepdims=False))

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
        Data points: {len(X_train[:,0]) + len(X_test[:,0])}; Method: Decision tree"
    )
    plt.savefig("plots/test24/test24_decision_tree_2")
    plt.show()
    plt.close()


"""
TEST 25:
DATASET: Housing data (regression case)
METHOD: Random forest

Bias-variance tradeoff. The error vs. the complexity of the method,
where the complexity in random forest is the depth of the trees. The amount of
trees in the random forest is set by default to 100. 
"""

test25 = False
if test25:
    print('>> RUNNING TEST 25:')
    n_components = 8
    m_observations = 2000
    X_train, X_test, y_train, y_test = helper.load_housing_california_data(
        n_components, m_observations)

    # Store the MSE, bias and variance values for test data
    list_of_MSE_testing = []
    list_of_bias_testing = []
    list_of_variance_testing = []

    iter = 0
    max_degree = 20
    n_bootstraps = 50
    # Finding MSE, bias and variance of test data for different degrees
    for degree in range(1, max_degree + 1):

        # Create matrix for storing the bootstrap results
        y_pred_test_matrix = np.empty((y_test.shape[0], n_bootstraps))

        # Running bootstrap-method
        for i in range(n_bootstraps):

            _X, _y, = helper.resample(X_train, y_train)

            # Create randomforest instance, with amount of max_depth
            clf = RandomForestRegressor(max_depth=degree)

            # Fit the data to the model we have created
            clf.fit(_X, _y)

            # Make predictions
            y_pred_test_matrix[:, i] = clf.predict(X_test).reshape(1, -1)

        iter += 1
        print(f"---------------- {iter}/{max_degree} RUNS ----------------")

        # Finding MSE, bias and variance from the bootstra
        MSE_test = np.mean(np.mean(
            (y_pred_test_matrix - np.transpose(np.array([y_test])))**2))

        bias_test = np.mean(
            (y_test - np.mean(y_pred_test_matrix, axis=1, keepdims=False))**2)

        variance_test = np.mean(
            np.var(y_pred_test_matrix, axis=1, keepdims=False))

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
        Data points: {len(X_train[:,0]) + len(X_test[:,0])}; Method: Random forest"
    )
    plt.savefig('plots/test25/test25_random_forest_1')
    plt.show()
    plt.close()
