from math import log
import time
from FF_Neural_Network import Neural_Network
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from cost_functions import logistic_cost
import pandas as pd
import autograd.numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper

# TODO: maybe delete this function
# TODO: We need to use the accuracy score when looking at the classification problem


# TODO: maybe delete - or make it clear that this was for own testing
def test_cancer_data(n_components: int = 2):
    """
    :param n_components (int):
        the n most important features of the breast cancer data.
    """

    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = helper.load_cancer_data(
        n_components)

    no_hidden_nodes = 2
    no_hidden_layers = 1

    FFNN = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=1,
        no_hidden_nodes=no_hidden_nodes,
        no_hidden_layers=no_hidden_layers
    )
    FFNN.set_SGD_values(
        n_epochs=20,
        batch_size=10,
        gamma=0.8,
        eta=0.01,
        lmbda=1e-5)
    FFNN.set_cost_function(logistic_cost)

    FFNN.set_activation_function_output_layer('sigmoid')
    FFNN.train_model(X_cancer_train, y_cancer_train)
    # FFNN.plot_MSE_of_last_training()
    # FFNN.plot_accuracy_score_last_training()
    FFNN.set_activation_function_output_layer('sigmoid_classification')
    print(
        f'Accuracy_train = {accuracy_score(FFNN.feed_forward(X_cancer_train),  y_cancer_train)}')
    print(
        f'Accuracy_test = {accuracy_score(FFNN.feed_forward(X_cancer_test),  y_cancer_test)}')

    return

    learning_rates = np.logspace(0, -5, 6)
    lmbda_values = np.logspace(0, -5, 6)

    train_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))
    test_accuracy_score = np.zeros((len(learning_rates), len(lmbda_values)))
    FFNN1 = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=1,
        no_hidden_nodes=no_hidden_nodes,
        no_hidden_layers=no_hidden_layers
    )
    FFNN1.set_SGD_values(
        n_epochs=10,
        batch_size=100,
        gamma=0.8)

    for i, eta in enumerate(learning_rates):
        for j, lmbda in enumerate(lmbda_values):
            FFNN1.set_activation_function_output_layer('sigmoid')
            FFNN1.refresh_the_biases()
            FFNN1.refresh_the_weights()
            # TODO: default values shall not be there, set the default values in the __init__ method instead
            FFNN1.set_SGD_values(
                eta=eta,
                lmbda=lmbda,
                n_epochs=20,
                gamma=0.8,
                batch_size=10
            )
            # FFNN1.train_model(X_cancer_train, y_cancer_train)
            FFNN1.train_model(X_cancer_train, y_cancer_train)
            FFNN1.set_activation_function_output_layer(
                'sigmoid_classification')
            train_accuracy_score[i][j] = helper.accuracy_score(
                y_cancer_train, FFNN1.feed_forward(X_cancer_train))
            test_accuracy_score[i][j] = helper.accuracy_score(
                y_cancer_test, FFNN1.feed_forward(X_cancer_test))

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(train_accuracy_score, annot=True, ax=ax, cmap="viridis",
                xticklabels=lmbda_values, yticklabels=learning_rates)

    # TODO: remove the save fig
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig("BREAST_heatmap_training_4.png")
    plt.show()
    print("Plotten er klar!!!")

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(test_accuracy_score, annot=True, ax=ax, cmap="viridis",
                xticklabels=lmbda_values, yticklabels=learning_rates)
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig("BREAST_heatmap_testing_4.png")
    plt.show()

    exit()

    FFNN.set_activation_function_output_layer("sigmoid_classification")


def main():

    test_cancer_data()


if __name__ == '__main__':
    a = time.time()
    main()
    print('TID: ', end='')
    print(time.time() - a)
