from FF_Neural_Network import Neural_Network
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper

# TODO: We need to use the accuracy score when looking at the classification problem


def test_cancer_data(n_components: int = 2):
    """

    :param n_components (int):
        the n most important features of the breast cancer data.
    """

    cancer = load_breast_cancer()

    # Parameter labels
    labels = cancer.feature_names[0:30]

    # 569 rows (sample data), 30 columns (parameters)
    X_cancer = cancer.data
    # 569 rows (0 for benign and 1 for malignant)
    y_cancer = cancer.target
    y_cancer = y_cancer.reshape(-1, 1)

    pca = PCA(n_components=n_components)
    X_cancer_2D = pca.fit_transform(X_cancer)

    X_scalar = 1/np.max(X_cancer_2D)
    X_cancer_2D_scaled = X_cancer_2D*X_scalar

    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = helper.train_test_split(
        X_cancer_2D_scaled, y_cancer)

    no_hidden_nodes = 50
    no_hidden_layers = 7

    FFNN = Neural_Network(
        no_input_nodes=n_components,
        no_output_nodes=1,
        no_hidden_nodes=no_hidden_nodes,
        no_hidden_layers=no_hidden_layers
    )
    FFNN.set_SGD_values(
        n_epochs=20,
        batch_size=100,
        gamma=0.8,
        eta=0.01)

    for i in range(1, 10):
        FFNN.set_activation_function_output_layer('sigmoid')
        FFNN.train_model(X_cancer_2D_scaled, y_cancer)
        # FFNN.plot_accuracy_score_last_training()
        FFNN.set_activation_function_output_layer('sigmoid_classification')
        print(
            f'Accuracy (after {i} training): = {accuracy_score(FFNN.feed_forward(X_cancer_2D_scaled),  y_cancer)}')

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
    plt.savefig("BREAST_heatmap_training_3.png")
    plt.show()
    print("Plotten er klar!!!")

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(test_accuracy_score, annot=True, ax=ax, cmap="viridis",
                xticklabels=lmbda_values, yticklabels=learning_rates)
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.savefig("BREAST_heatmap_testing_3.png")
    plt.show()

    exit()

    FFNN.set_activation_function_output_layer("sigmoid_classification")

    exit()
    print(pca.explained_variance_ratio_)
    print(pca.components_)
    print(pd.DataFrame(pca.components_,
                       columns=X_cancer[0, :], index=['PC-1', 'PC-2']))

    exit()
    print("Eigenvector of largest eigenvalue")
    print(pca.components_.T[:, 0])


def main():

    test_cancer_data()


if __name__ == '__main__':
    main()
