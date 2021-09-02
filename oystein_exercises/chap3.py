"""
Exercise 2:
making your own data and exploring scikit-learn
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


# TODO: write code computing the parametrization of the data set fitting a second-order polynomial


def polynomial_fit_n_order(x: np.ndarray, y: np.ndarray, n: int = 2):
    """
    Function for creating a polynomial of degree n to fit the given data

    :param x:
    :param y:
    :param n:

    :return:
        None
    source: https://moonbooks.org/Articles/How-to-implement-a-polynomial-linear-regression-using-scikit-learn-and-python-3-/
    """

    # Step 1: (source: training data)
    X = x[:, np.newaxis]  # Convert from vector to a matrix, n*1 matrix
    Y = y[:, np.newaxis]  # Convert from vector to a matrix, n*1 matrix
    plt.scatter(X, Y, label="datapoints")   # Plotting the datapoints

    # Step 2: (source: data preparation)
    polynomial_features = PolynomialFeatures(degree=n)

    """
    X_transformed's form for n-th order polynomial (MATRIX), x => m*1-matrix:
    1 x_11 x_11**2 ... x_11**n
    1 x_21 x_21**2 ... x_21**n
    .                   .
    .                   .
    .                   .
    1 x_m1 x_m1**2 ... x_m1**n

    X_transformed's form for 2-th order polynomial (MATRIX), x => m*2-matrix:
    1 x_11 x_12 x_11**2 x_11*x_12 x_22**2
    ...
    """
    X_transformed = polynomial_features.fit_transform(X)

    # Step 3: (source: define and train a model)
    the_model = LinearRegression().fit(X_transformed, Y)

    # Step 4: (source: calculate bias and variance)
    Y_predicted = the_model.predict(X_transformed)

    rmse = np.sqrt(mean_squared_error(Y, Y_predicted))
    r2 = r2_score(Y, Y_predicted)
    mse = 1/n * np.sum((Y-Y_predicted)**2)

    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'R2: {r2}')

    # Step 5: (source: prediction)
    x_min = -1
    x_max = 1.2
    x_axis_span = np.linspace(x_min, x_max, 100)

    x_axis_span = x_axis_span[:, np.newaxis]    # Vector to matrix

    X_transformed_correct_span = polynomial_features.fit_transform(x_axis_span)
    Y_final_predict = the_model.predict(X_transformed_correct_span)

    title_for_plot = f"""Polynomial Linear Regression using scikit-learn
            Degree = {n}; RMSE = {rmse: .2}; R2 = {r2: .2}"""

    # plt.plot(x_axis_span, Y_final_predict,
    #          label=f"Prediction line", linewidth=2)
    # plt.grid()
    # plt.xlim(x_min, x_max)
    # plt.ylim(0, 10)
    # plt.title(title_for_plot, fontsize=10)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.savefig("polynomial_linear_regression.png", bbox_inches='tight')
    # plt.legend()
    # plt.show()


def polynomial_fit_n_order_without_packages(x: np.ndarray, y: np.ndarray, n: int = 2):
    X = x[:, np.newaxis]  # Convert from vector to a matrix, n*1 matrix
    Y = y[:, np.newaxis]  # Convert from vector to a matrix, n*1 matrix

    X_transformed = np.zeros([len(X), n+1])
    for i in range(len(X)):
        for j in range(n+1):
            X_transformed[i, j] = X[i]**j

    # TODO: create this without the LinearRegression()-function
    the_model = LinearRegression().fit(X_transformed, Y)

    # TODO: Check if the_model and Beta are the same
    Beta = matrix_inverse(
        X_transformed.T @ X_transformed) @ X_transformed.T @ Y

    print(Beta)

    x_axis_span = np.linspace(-1, 1.2, 100)[:, np.newaxis]
    X_axis_span = np.zeros([len(x_axis_span), n+1])

    for i in range(len(x_axis_span)):
        for j in range(n+1):
            X_axis_span[i, j] = x_axis_span[i]**j

    Y_predicted = X_axis_span @ Beta

    # plt.scatter(X, Y, label="datapoints")   # Plotting the datapoints
    # plt.plot(x_axis_span, Y_predicted, label="predicted line")
    # plt.grid()
    # plt.legend()
    # plt.show()


def matrix_multiplication(matrix1: np.ndarray, matrix2: np.ndarray):
    return np.matmul(matrix1, matrix2)


def matrix_inverse(matrix):
    return np.linalg.inv(matrix)


def matrix_transpose(matrix: np.ndarray):
    return matrix.transpose()


if __name__ == '__main__':
    observations = 200
    x = np.random.rand(observations)  # rows = 100, columns = 1
    y = 2 + 5*x**2 + 0.1 * np.random.randn(observations)
    # x_perfect = np.linspace(-100, 100, 100000)
    # plt.plot(x_perfect, 2 + 5*x_perfect**2,
    #          linestyle="dotted", label="Exact")

    # if len(sys.argv) > 1:
    #     try:
    #         polynomial_fit_n_order(x, y, int(sys.argv[1]))
    #     except Exception:
    #         polynomial_fit_n_order(x, y)

    # else:
    # polynomial_fit_n_order(x, y)

    polynomial_fit_n_order_without_packages(x, y)
