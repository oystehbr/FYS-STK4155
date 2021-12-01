"""
This is a helper function. These functions are used in several
exercises, and we chose to make this to improve structure
"""
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer, load_iris, fetch_california_housing, load_diabetes
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.core import Activation


def franke_function(x: float, y: float):
    """
    Compute and return function value for a Franke's function

    :param x (float):
        input value
    :param y (float):
        input value

    :return (float):
        function value
    """

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def generate_data(n: int, noise_multiplier: float = 0.1):
    """
    Generate n data points for x and y, and calculated z
    with Franke function

    :param n (int):
        number of x and y values
    :param noise_multiplier (float, int):
        scale the noise

    :return tuple[np.ndarray, np.ndarray, np.ndarray]:
        - the (random) generated x values
        - the (random) generated y values
        - the output value from Franke function
    """

    data_array = np.zeros(n)
    x_array = np.zeros(n)
    y_array = np.zeros(n)
    for i in range(n):
        x_array[i] = np.random.uniform(0, 1)
        y_array[i] = np.random.uniform(0, 1)
        eps = np.random.normal(0, 1)
        data_array[i] = franke_function(
            x_array[i], y_array[i]) + noise_multiplier * eps

    return x_array, y_array, data_array


def create_design_matrix(x, y, degree: int):
    """
    Function for creating and returning a
    design matrix for a given degree.

    :param x (np.ndarray):
        a dependent variable for the design matrix
    :param y (np.ndarray):
        a dependent variable for the design matrix
    :param degree (int):
        the order of the polynomial that defines the design matrix

    :return (np.ndarray):
        the design matrix
    """

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    col_no = len(x)
    row_no = int((degree+1)*(degree+2)/2)  # Number of elements in beta
    X = np.ones((col_no, row_no))

    for i in range(1, degree+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:, q+k] = (x**(i-k))*(y**k)

    return X


def get_beta_OLS(X, z_values):
    """
    Function for creating and return the regression parameters,
    beta, by using the regression method: OLS

    :param X (np.ndarray):
        a design matrix
    :param z_values (np.ndarray):
        the response variable

    :return (np.ndarray):
        the regression parameters, beta
    """

    X_T = np.matrix.transpose(X)
    beta = np.linalg.pinv(X_T @ X) @ X_T @ z_values

    return beta


def get_beta_RIDGE(X, z_values, lmbda: float):
    """
    Function for creating and return the regression parameters,
    beta, by using the regression method: Ridge

    :param X (np.ndarray):
        a design matrix
    :param z_values (np.ndarray):
        the response variable
    :param lmbda (float):
        parameter used by Ridge regression (lambda)

    :return (np.ndarray):
        the regression parameters, beta
    """

    X_T = np.matrix.transpose(X)
    p = X.shape[1]
    I = np.eye(p, p)

    beta = np.linalg.pinv(X_T @ X + lmbda*I) @ X_T @ z_values

    return beta


def get_beta_LASSO(X, z_values, lmbda: float):
    """
    Function for creating and return the regression parameters,
    beta, by using the regression method: Lasso

    :param X (np.ndarray):
        a design matrix
    :param z_values (np.ndarray):
        the response variable
    :param lmbda (float):
        parameter used by Lasso regression (lambda)

    :return (np.ndarray):
        the regression parameters, beta
    """

    model_lasso = Lasso(lmbda, fit_intercept=False)
    model_lasso.fit(X, z_values)
    beta = model_lasso.coef_

    return beta


def predict_output(x_train, y_train, z_train, x_test, y_test, degree: int, regression_method: str = 'OLS', lmbda: float = 1):
    """
    The function creates a model (with regression method as preffered)
    with respect to the training data (with scaling). Then it will predict
    the outcome to our test- and training set and return the predictions
    and the regression parameters that were used in the model

    :param x_train (np.ndarray):
        training data: a dependent variable for the design matrix
    :param y_train (np.ndarray):
        training data: a dependent variable for the design matrix
    :param z_train (np.ndarray):
        training data: a response variable for the beta prediction
    :param x_test (np.ndarray):
        testing data: a dependent variable for the testing design matrix
    :param y_test (np.ndarray):
        testing data: a dependent variable for the testing design matrix
    :param degree (int):
        the order of the polynomial that defines the design matrix
    :param regression_method (str):
        the preffered regression method: OLS, RIDGE or LASSO
    :param lmbda (float):
        parameter used by Ridge and Lasso regression (lambda)

    :return (tuple[np.array, np.array, np.array]):
        - predicted outcome of the testing set (with our model)
        - predicted outcome of the training set (with our model)
        - the regression parameters, used in our model
    """

    # Get designmatrix from the training data and scale it
    X_train = create_design_matrix(x_train, y_train, degree)
    X_train_scale = np.mean(X_train, axis=0)
    X_train_scaled = X_train - X_train_scale

    # Get designmatrix from the test data and scale it
    X_test = create_design_matrix(x_test, y_test, degree)
    X_test_scaled = X_test - X_train_scale

    # Scale the output_values
    z_train_scale = np.mean(z_train, axis=0)
    z_train_scaled = z_train - z_train_scale

    # Get the beta from the given method
    if regression_method == 'OLS':
        beta = get_beta_OLS(X_train_scaled, z_train_scaled)
    elif regression_method == 'RIDGE':
        beta = get_beta_RIDGE(X_train_scaled, z_train_scaled, lmbda)
    elif regression_method == 'LASSO':
        beta = get_beta_LASSO(X_train_scaled, z_train_scaled, lmbda)
    else:
        raise Exception(f"Incorrect regression method: {regression_method}")

    # Find out the prediction on our known data (which was not including in training)
    # And scaling it back to its original form
    z_pred_test = (X_test_scaled @ beta) + z_train_scale

    # Find out how good the model is on our training data
    z_pred_train = (X_train_scaled @ beta) + z_train_scale

    return z_pred_test, z_pred_train, beta


def seaborn_plot_lmbda_learning(score, x_tics, y_tics, score_name, save_name=None):
    """
    Seaborn plot of the combination of lambda and eta values will be
    shown and saved if save_name is provided.

    :param score (list[list]):
        the scores of lambda and eta values
    :param x_tics (np.ndarray):
        the lmbda values that were used
    :param y_tics (np.ndarray):
        the eta values that were used
    :param score_name (str):
        name of the score-evaluation, for plotting title.
    :param save_name (str), default = None:
        the name of the plot, for saving (including .png)
    """

    sns.set()
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(score, annot=True, ax=ax, cmap="viridis",
                xticklabels=x_tics, yticklabels=y_tics)

    ax.set_title(f'{score_name}')
    ax.set_xlabel("$\lambda$")
    ax.set_ylabel("$\eta$")

    if save_name != None:
        plt.savefig(save_name)
    plt.show()


def seaborn_plot_architecture(score, x_tics, y_tics, score_name, save_name=None):
    """
    Seaborn plot of the combination of hidden layers and nodes as the architecture of
    the neural network will be shown and saved if save_name is provided.

    :param score (list[list]):
        the scores of lambda and eta values
    :param x_tics (np.ndarray):
        the hidden layers values used
    :param y_tics (np.ndarray):
        the hidden nodes values used
    :param score_name (str):
        name of the score-evaluation, for plotting title.
    :param save_name (str), default = None:
        the name of the plot, for saving (including .png)
    """
    sns.set()
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(score, annot=True, ax=ax, cmap="viridis",
                xticklabels=x_tics, yticklabels=y_tics)

    ax.set_title(f'{score_name}')
    ax.set_xlabel("#layers")
    ax.set_ylabel("#nodes")

    if save_name != None:
        plt.savefig(save_name)
    plt.show()


def seaborn_plot_batchsize_gamma(score, x_tics, y_tics, score_name, save_name=None):
    """
    Seaborn plot of the combination of batch size and the momentum parameter
    gamma will be shown and saved if save_name is provided.

    :param score (list[list]):
        the scores of lambda and eta values
    :param x_tics (np.ndarray):
        the gamma values that were used
    :param y_tics (np.ndarray):
        the batchsize values that were used
    :param score_name (str):
        name of the score-evaluation, for plotting title.
    :param save_name (str), default = None:
        the name of the plot, for saving (including .png)
    """

    sns.set()
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(score, annot=True, ax=ax, cmap="viridis",
                xticklabels=x_tics, yticklabels=y_tics)

    ax.set_title(f'{score_name}')
    ax.set_xlabel("gamma")
    ax.set_ylabel("batch_size")

    if save_name != None:
        plt.savefig(save_name)
    plt.show()


def seaborn_plot_no_minibatches_eta(score, x_tics, y_tics, score_name, save_name=None):
    """
    Seaborn plot of the combination of the number of minibatches and eta values will be
    showned and saved if save_name is provided.

    :param score (list[list]):
        the scores of lambda and eta values
    :param x_tics (np.ndarray):
        learning rates, eta
    :param y_tics (np.ndarray):
        number of minibatches
    :param score_name (str):
        name of the score-evaluation, for plotting title.
    :param save_name (str), default = None:
        the name of the plot, for saving (including .png)
    """

    sns.set()
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(score, annot=True, ax=ax, cmap="viridis",
                xticklabels=x_tics, yticklabels=y_tics)

    ax.set_title(f'{score_name}')
    ax.set_xlabel("$\eta$")
    ax.set_ylabel("number of minibatches")

    if save_name != None:
        plt.savefig(save_name)
    plt.show()


def seaborn_plot_max_depth_and_min_sample_leaf(score, x_tics, y_tics, score_name, save_name=None):
    """
    Seaborn plot of the combination of lambda and eta values will be
    shown and saved if save_name is provided.

    :param score (list[list]):
        the scores of lambda and eta values
    :param x_tics (np.ndarray):
        the min_sample_leaf values that were used
    :param y_tics (np.ndarray):
        the max_depth values that were used
    :param score_name (str):
        name of the score-evaluation, for plotting title.
    :param save_name (str), default = None:
        the name of the plot, for saving (including .png)
    """

    sns.set()
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(score, annot=True, ax=ax, cmap="viridis",
                xticklabels=x_tics, yticklabels=y_tics)

    ax.set_title(f'{score_name}')
    ax.set_xlabel("min sample leaf")
    ax.set_ylabel("max depth")

    if save_name != None:
        plt.savefig(save_name)
    plt.show()


def load_cancer_data(n):
    """
    Loading the cancer_data from scikit-learn. Selecting the
    features with the highest explained variance ratio (the first
    # n) and returning those

    :param n (int):
        amount of components we want to return

    :return tuple(np.ndarray):
        - Input data, training
        - Input data, testing
        - Target data, training
        - Target data, testing
    """

    # Loading cancer data
    cancer = load_breast_cancer()
    # Parameter labels (if you want, not used)
    labels = cancer.feature_names[0:30]

    X_cancer = cancer.data
    y_cancer = cancer.target    # 0 for benign and 1 for malignant
    y_cancer = y_cancer.reshape(-1, 1)

    # Selecting the n first components w.r.t. the PCA
    pca = PCA(n_components=n)
    X_cancer_nD = pca.fit_transform(X_cancer)

    X_cancer_nD = X_cancer_nD/(X_cancer_nD.max(axis=0))

    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(
        X_cancer_nD, y_cancer)

    return X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test

    print(pca.explained_variance_ratio_)


def load_iris_data(n, show_explained_ratio=False):
    """
    Loading the cancer_data from scikit-learn. Selecting the
    features with the highest explained variance ratio (the first
    # n) and returning those

    :param n (int):
        amount of components we want to return

    :return tuple(np.ndarray):
        - Input data, training
        - Input data, testing
        - Target data, training
        - Target data, testing
    """

    # Loading cancer data
    pd = load_iris()
    # Parameter labels (if you want, not used)

    X_input = pd.data
    y_target = pd.target    # 0 for benign and 1 for malignant
    y_target = y_target.reshape(-1, 1)

    # Shuffling the data
    values = np.concatenate((X_input, y_target), axis=1)
    np.random.shuffle(values)

    X_input = values[:, :-1]
    y_target = values[:, -1]

    # Selecting the n first components w.r.t. the PCA
    pca = PCA(n_components=n)
    X_nD = pca.fit_transform(X_input)

    if show_explained_ratio:
        print(
            f'Total explained variance ratio (of {n} component): {sum(pca.explained_variance_ratio_)}')

    X_nD = X_nD/(X_nD.max(axis=0))

    X_train, X_test, y_train, y_test = train_test_split(
        X_nD, y_target)

    return X_train, X_test, y_train, y_test


def load_diabetes_PSY_data(n, show_explained_ratio=False):
    """
    Loading the cancer_data from scikit-learn. Selecting the
    features with the highest explained variance ratio (the first
    # n) and returning those

    :param n (int):
        amount of components we want to return

    :return tuple(np.ndarray):
        - Input data, training
        - Input data, testing
        - Target data, training
        - Target data, testing
    """

    # Loading cancer data
    diabetes = load_diabetes()
    # Parameter labels (if you want, not used)

    # TODO: shuffle the data
    X_input = diabetes.data
    y_target = diabetes.target    # 0 for benign and 1 for malignant
    y_target = y_target.reshape(-1, 1)

    values = np.concatenate((X_input, y_target), axis=1)
    np.random.shuffle(values)

    # Selecting the n first components w.r.t. the PCA
    X_input = values[:, :-1]
    y_target = values[:, -1]

    # Selecting the n first components w.r.t. the PCA
    pca = PCA(n_components=n)
    X_nD = pca.fit_transform(X_input)

    if show_explained_ratio:
        print(
            f'Total explained variance ratio (of {n} component): {sum(pca.explained_variance_ratio_)}')

    # X_train_scale = np.mean(X_train, axis=0)
    X_nD = (X_nD - np.mean(X_nD, axis=0))/np.std(X_nD, axis=0)
    # X_nD = X_nD/(X_nD.max(axis=0))

    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(
        X_nD, y_target)

    return X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test


def load_housing_california_data(n, m_observations=1000, show_explained_ratio=False):
    """
    Loading the cancer_data from scikit-learn. Selecting the
    features with the highest explained variance ratio (the first
    # n) and returning those

    :param n (int):
        amount of components we want to return

    :return tuple(np.ndarray):
        - Input data, training
        - Input data, testing
        - Target data, training
        - Target data, testing
    """

    # Loading cancer data
    pd = fetch_california_housing()
    # Parameter labels (if you want, not used)

    X_input = pd.data  # [:1000]
    y_target = pd.target  # [:1000]
    y_target = y_target.reshape(-1, 1)

    # Shuffling the data
    values = np.concatenate((X_input, y_target), axis=1)
    np.random.shuffle(values)
    X_input = values[:m_observations, :-1]
    y_target = values[:m_observations, -1]

    # # Selecting the n first components w.r.t. the PCA
    pca = PCA(n_components=n)
    X_nD = pca.fit_transform(X_input)

    if show_explained_ratio:
        print(
            f'Total explained variance ratio (of {n} component): {sum(pca.explained_variance_ratio_)}')

    X_nD = X_nD/(X_nD.max(axis=0))

    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(
        X_nD, y_target)

    return X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test


def load_diabetes_data(n_components, m_observations=1000, show_explained_ratio=False):
    """
    # TODO: docstrings
    """

    path = "data/diabetes_012_health_indicators_BRFSS2015.csv"
    # Loading the data into pandas
    df = pd.read_csv(
        path,
        sep=","
    )

    diabetes_values = df.values

    # Shuffle the data
    np.random.shuffle(diabetes_values)

    X_input = diabetes_values[:m_observations, 1:]
    y_target = diabetes_values[:m_observations, 0]

    pca = PCA(n_components)

    # Only include the "n_components" most important features
    X_input_PCA = pca.fit_transform(X_input)

    if show_explained_ratio:
        print(pca.explained_variance_ratio_)

    X_input_PCA = X_input_PCA/(X_input_PCA.max(axis=0))

    X_train, X_test, y_train, y_test = train_test_split(
        X_input_PCA, y_target)

    X_train, y_train = midsampling_of_training_data(X_train, y_train)

    return X_train, X_test, y_train, y_test


def load_dry_beans_data(n_components, m_observations=1000, show_explained_ratio=False, show_target_distribution=False):
    """
    # TODO: docstrings
    """

    path = "data/Dry_Bean_Dataset.xlsx"
    # Loading the data into pandas
    df = pd.read_excel(path, index_col=0)

    values = df.values
    for i in range(len(values)):
        if values[i, -1] == 'SEKER':
            values[i, -1] = 0
        elif values[i, -1] == 'BARBUNYA':
            values[i, -1] = 1
        elif values[i, -1] == 'BOMBAY':
            values[i, -1] = 2
        elif values[i, -1] == 'CALI':
            values[i, -1] = 3
        elif values[i, -1] == 'HOROZ':
            values[i, -1] = 4
        elif values[i, -1] == 'SIRA':
            values[i, -1] = 5
        elif values[i, -1] == 'DERMASON':
            values[i, -1] = 6

    # Shuffle the data
    np.random.shuffle(values)

    if show_target_distribution:
        sum_all = values.shape[0]
        print(">> The distribution of the targets ")
        print(f'0: {sum(values[:, -1] == 0)/sum_all : 3.3f} (SEKER)')
        print(f'1: {sum(values[:, -1] == 1)/sum_all : 3.3f} (BARBUNYA)')
        print(f'2: {sum(values[:, -1] == 2)/sum_all : 3.3f} (BOMBAY)')
        print(f'3: {sum(values[:, -1] == 3)/sum_all : 3.3f} (CALI)')
        print(f'4: {sum(values[:, -1] == 4)/sum_all : 3.3f} (HOROZ)')
        print(f'5: {sum(values[:, -1] == 5)/sum_all : 3.3f} (SIRA)')
        print(f'6: {sum(values[:, -1] == 6)/sum_all : 3.3f} (DERMASON)')


    X_input = values[:m_observations, :-1]
    y_target = values[:m_observations, -1]

    pca = PCA(n_components)

    # Only include the "n_components" most important features
    X_input_PCA = pca.fit_transform(X_input)

    if show_explained_ratio:
        print(
            f'Total explained variance ratio (of {n_components} component): {sum(pca.explained_variance_ratio_) : 3.3f}')

    X_input_PCA = X_input_PCA/(X_input_PCA.max(axis=0))

    X_train, X_test, y_train, y_test = train_test_split(
        X_input_PCA, y_target)

    return X_train, X_test, y_train, y_test


def load_wine_data(n_components, m_observations=1000, show_explained_ratio=False):
    """
    # TODO: docstrings
    """

    path = "data/wineQualityReds.csv"
    # Loading the data into pandas
    df = pd.read_csv(
        path,
        sep=","
    )

    values = df.values

    # Shuffle the data
    np.random.shuffle(values)

    X_input = values[:m_observations, :-1]
    y_target = values[:m_observations, -1]

    pca = PCA(n_components)

    # Only include the "n_components" most important features
    X_input_PCA = pca.fit_transform(X_input)

    if show_explained_ratio:
        print(pca.explained_variance_ratio_)
        print(sum(pca.explained_variance_ratio_))

    # X_input = X_input/(X_input.max(axis=0))
    X_input_PCA = X_input_PCA/(X_input_PCA.max(axis=0))

    X_train, X_test, y_train, y_test = train_test_split(
        X_input_PCA, y_target)

    # X_train, y_train = oversampling_of_training_data(X_train, y_train)

    return X_train, X_test, y_train, y_test


def load_muscle_data(n_components, m_observations=1000, show_explained_ratio=False):
    """
    # TODO: docstrings
    """

    path_0 = "data/muscle_data/0.csv"
    path_1 = "data/muscle_data/1.csv"
    path_2 = "data/muscle_data/2.csv"
    path_3 = "data/muscle_data/3.csv"
    # Loading the data into pandas
    values_0 = pd.read_csv(path_0, sep=",").values
    values_1 = pd.read_csv(path_1, sep=",").values
    values_2 = pd.read_csv(path_2, sep=",").values
    values_3 = pd.read_csv(path_3, sep=",").values

    values = np.append(values_0, values_1, axis=0)
    values = np.append(values, values_2, axis=0)
    values = np.append(values, values_3, axis=0)

    # Shuffle the data
    np.random.shuffle(values)

    X_input = values[:m_observations, :-1]
    y_target = values[:m_observations, -1]

    pca = PCA(n_components)

    # Only include the "n_components" most important features
    X_input_PCA = pca.fit_transform(X_input)

    if show_explained_ratio:
        print(pca.explained_variance_ratio_)
        print(
            f'sum of explained variance ratio {sum(pca.explained_variance_ratio_)}')

    # X_input_PCA = X_input_PCA/(X_input_PCA.max(axis=0))

    X_train, X_test, y_train, y_test = train_test_split(
        X_input_PCA, y_target)

    return X_train, X_test, y_train, y_test


def load_diabetes_data_without_PCA(n_components, m_observations=1000):
    """
    # TODO: docstrings
    https://towardsdatascience.com/having-an-imbalanced-dataset-here-is-how-you-can-solve-it-1640568947eb
    """

    path = "data/diabetes_012_health_indicators_BRFSS2015.csv"
    # Loading the data into pandas
    df = pd.read_csv(
        path,
        sep=","
    )

    diabetes_values = df.values

    # Shuffle the data
    np.random.shuffle(diabetes_values)

    X_input = diabetes_values[:m_observations, 1:]
    y_target = diabetes_values[:m_observations, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X_input, y_target)

    X_train, y_train = oversampling_of_training_data(X_train, y_train)

    return X_train, X_test, y_train, y_test


def load_air_data(n_components, m_observations=1000, show_explained_ratio=False):
    path = "data/AirQuality.csv"
    df = pd.read_csv(
        path,
        sep=";"
    )

    df = df.drop(['Date', 'Time', 'Unnamed: 15', 'Unnamed: 16'], axis=1)
    df = df.astype(float)

    values = df.values
    max = values.shape[0]
    i = 0
    while i < max:
        row = values[i]
        if -200 in row:
            values = np.delete(values, i, 0)
            max -= 1
        else:
            i += 1

    np.random.shuffle(values)

    X_input = values[:m_observations, :-1]
    y_target = values[:m_observations, -1]

    pca = PCA(n_components)

    # Only include the "n_components" most important features
    X_input_PCA = pca.fit_transform(X_input)

    if show_explained_ratio:
        print(pca.explained_variance_ratio_)
        print(sum(pca.explained_variance_ratio_))

    # X_input = X_input/(X_input.max(axis=0))
    X_input_PCA = X_input_PCA/(X_input_PCA.max(axis=0))

    X_train, X_test, y_train, y_test = train_test_split(
        X_input_PCA, y_target)

    return X_train, X_test, y_train, y_test


def convert_num_to_vec(y, dim):
    """
    # TODO:

    Dimensional of the converter
    """

    y_list = []
    for i, _y in enumerate(y):
        for j in range(dim):
            if _y == j:
                new_list = [0]*dim
                new_list[j] = 1
                y_list.append(new_list)

    return np.array(y_list)


def convert_vec_to_num(y):
    """
    # TODO:
    """

    y_list = list(y)
    for i, liste in enumerate(y_list):
        leader_amount = 0
        leader = 0
        for j, num in enumerate(liste):
            if num > leader_amount:
                leader = j
                leader_amount = num

        y_list[i] = leader

    return np.array(y_list).reshape(-1, 1)


def oversampling_of_training_data(X_train, y_train):
    """

    https://towardsdatascience.com/having-an-imbalanced-dataset-here-is-how-you-can-solve-it-1640568947eb
    """

    # Doing the imbalanced data approach, oversampling
    pd_train = pd.DataFrame(X_train)
    y_col = len(pd_train.columns)
    pd_train.insert(y_col, y_col, y_train, True)

    pd_train_duplicates_0 = pd_train[y_train == 0]
    pd_train_duplicates_1 = pd_train[y_train == 1]
    pd_train_duplicates_2 = pd_train[y_train == 2]

    # Finding the maximum amount similar observations on target value
    max_amount = len(pd_train_duplicates_0)
    if len(pd_train_duplicates_1) > max_amount:
        max_amount = len(pd_train_duplicates_1)

    if len(pd_train_duplicates_2) > max_amount:
        max_amount = len(pd_train_duplicates_2)

    # Reproducing the occurances that are few
    for i in range(int(max_amount/len(pd_train_duplicates_0)) - 1):
        pd_train = pd.concat([pd_train, pd_train_duplicates_0])

    for i in range(int(max_amount / len(pd_train_duplicates_1)) - 1):
        pd_train = pd.concat([pd_train, pd_train_duplicates_1])

    for i in range(int(max_amount / len(pd_train_duplicates_2)) - 1):
        pd_train = pd.concat([pd_train, pd_train_duplicates_2])

    pd_train_values = pd_train.values
    # TODO: check if shuffling row-wise
    np.random.shuffle(pd_train_values)

    X_train = pd_train_values[:, :-1]
    y_train = pd_train_values[:, -1]

    return X_train, y_train


def undersampling_of_training_data(X_train, y_train):
    """[summary]
    Given that class 0 has most data
    """

    # Doing the imbalanced data approach, oversampling
    pd_train = pd.DataFrame(X_train)
    y_col = len(pd_train.columns)
    pd_train.insert(y_col, y_col, y_train, True)

    pd_train_duplicates_0 = pd_train[y_train == 0]
    pd_train_duplicates_1 = pd_train[y_train == 1]
    pd_train_duplicates_2 = pd_train[y_train == 2]

    if len(pd_train_duplicates_1) > len(pd_train_duplicates_2):
        pd_train_new_0 = pd_train_duplicates_0.sample(
            n=int(len(pd_train_duplicates_1)))
    else:
        pd_train_new_0 = pd_train_duplicates_0.sample(
            n=int(len(pd_train_duplicates_2)))

    pd_train = pd.concat(
        [pd_train_new_0, pd_train_duplicates_1, pd_train_duplicates_2])

    pd_train_values = pd_train.values
    np.random.shuffle(pd_train_values)

    X_train = pd_train_values[:, :-1]
    y_train = pd_train_values[:, -1]

    return X_train, y_train


def midsampling_of_training_data(X_train, y_train):
    """
    # TODO: docstrings
    """

    pd_train = pd.DataFrame(X_train)
    y_col = len(pd_train.columns)
    pd_train.insert(y_col, y_col, y_train, True)

    pd_train_duplicates_0 = pd_train[y_train == 0]
    pd_train_duplicates_1 = pd_train[y_train == 1]
    pd_train_duplicates_2 = pd_train[y_train == 2]

    # Undersampling of 0 (double of 2)
    pd_train_new_0 = pd_train_duplicates_0.sample(
        n=int(2 * len(pd_train_duplicates_2)))

    pd_train = pd.concat(
        [pd_train_new_0, pd_train_duplicates_1, pd_train_duplicates_2])

    # Oversampling of 1 (half of 2)
    for i in range(int(len(pd_train_duplicates_2) / len(pd_train_duplicates_1)/2)):
        pd_train = pd.concat([pd_train, pd_train_duplicates_1])

    pd_train_values = pd_train.values
    np.random.shuffle(pd_train_values)

    X_train = pd_train_values[:, :-1]
    y_train = pd_train_values[:, -1]

    return X_train, y_train


def create_NN(input_nodes, hidden_nodes, hidden_layers, loss, act_func, eta, lmbda, gamma):
    """
    # TODO: docstrings 

    """

    model = Sequential()
    for degree in range(hidden_layers):
        if degree == 0:
            model.add(Dense(60, activation=act_func, kernel_regularizer=regularizers.l2(
                lmbda), input_dim=input_nodes))
        else:
            model.add(Dense(60, activation=act_func,
                            kernel_regularizer=regularizers.l2(lmbda)))
    if loss == "mse":
        model.add(Dense(1))
    else:
        model.add(Dense(7, activation='softmax'))

    sgd = optimizers.SGD(learning_rate=eta, momentum=gamma)

    model.compile(loss=loss,
                  optimizer=sgd)

    return model




def main():
    a = np.array([[0.33992657, 0.34396532, 0.31610811]])
    b = np.array([[0.3238535,  0.33098534, 0.34516116]])

    res = convert_vec_to_num(a)
    print(res)

    # load_diabetes_data_without_PCA(2)


if __name__ == '__main__':
    main()
