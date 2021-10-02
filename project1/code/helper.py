"""
This is a helper function. These functions are used in several
exercises, and we chose to make this to improve structure
"""
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from numba import njit


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

    # TODO: vectorize
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


@njit
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

    # TODO: maybe explain this function
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

    model_lasso = Lasso(lmbda)
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
