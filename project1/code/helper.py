"""
All valuable functions will be added here
"""
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from numba import njit

# TODO: regression parameters is beta


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
        array of generated funciton values with noise
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

    if len(x.shape) > 1:
        x = np.ravel(x)  # TODO: hva gjør den
        y = np.ravel(y)

    col_no = len(x)
    row_no = int((degree+1)*(degree+2)/2)		# Number of elements in beta
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


def get_betas_RIDGE(X, z_values, lmbda: float):
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

    betas = np.linalg.pinv(X_T @ X + lmbda*I) @ X_T @ z_values

    return betas


def get_betas_LASSO(X, z_values, lmbda: float):
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
    betas = model_lasso.coef_

    return betas


def predict_output(x_train, y_train, z_train, x_test, y_test, degree, regression_method='OLS', lmbda=1):
    """
    The function takes in the training data and will create a model,
    with this model it will be scaled with # TODO: scale_method and
    will predict z

    :param x_train ():
    :param y_train ():
    :param z_train ():
    :param regression_method (str):
    :param scale_method (str):

    # TODO: correct types
    :return (tuple(np.array, np.array)):

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

    # Get the betas from the given method
    if regression_method == 'OLS':
        betas = get_betas_OLS(X_train_scaled, z_train_scaled)
    elif regression_method == 'RIDGE':
        betas = get_betas_RIDGE(X_train_scaled, z_train_scaled, lmbda)
    elif regression_method == 'LASSO':
        betas = get_betas_LASSO(X_train_scaled, z_train_scaled, lmbda)
    else:
        # TODO: raise Error
        print("incorrect regression model in bias_variance_boot")

    # Find out the prediction on our known data (which was not including in training)
    # And scaling it back to its original form
    z_pred_test = (X_test_scaled @ betas) + z_train_scale

    # Find out how good the model is on our training data
    z_pred_train = (X_train_scaled @ betas) + z_train_scale

    return z_pred_test, z_pred_train, betas
