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

    Source: https://moonbooks.org/Articles/How-to-implement-a-polynomial-linear-regression-using-scikit-learn-and-python-3-/

    """

    # Step 1: training data x, y - input
    X = x[:, np.newaxis]
    Y = y[:, np.newaxis]

    plt.scatter(X, Y)

    # Step 2: data preparation
    polynomial_features = PolynomialFeatures(degree=n)
    X_transformed = polynomial_features.fit_transform(x[:, np.newaxis])

    # Step 3: define and train a model
    the_model = LinearRegression()
    the_model.fit(X_transformed, Y)

    # Step 4: calculate bias and variance
    Y_predicted = the_model.predict(X_transformed)

    rmse = np.sqrt(mean_squared_error(Y, Y_predicted))
    r2 = r2_score(Y, Y_predicted)

    print(f'RMSE: {rmse}')
    print(f'R2: {r2}')

    # Step 5: prediction
    x_min = -1
    x_max = 1.2
    x_axis_span = np.linspace(x_min, x_max, 100)

    # Increasing the dimension of the existing array
    x_axis_span = x_axis_span[:, np.newaxis]

    X_final_transform = polynomial_features.fit_transform(x_axis_span)
    Y_final_predict = the_model.predict(X_final_transform)

    title_for_plot = f"""Polynomial Linear Regression using scikit-learn
            Degree = {n}; RMSE = {rmse: .2}; R2 = {r2: .2}"""

    plt.plot(x_axis_span, Y_final_predict, color='coral',
             label="Prediction line", linewidth=3)
    plt.grid()
    plt.xlim(x_min, x_max)
    plt.ylim(0, 10)
    plt.title(title_for_plot, fontsize=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("polynomial_linear_regression.png", bbox_inches='tight')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    observations = 100
    x = np.random.rand(observations)  # rows = 100, columns = 1
    y = 2 + 5*x**2 + 0.1 * np.random.randn(observations)

    if len(sys.argv) > 1:
        try:
            polynomial_fit_n_order(x, y, int(sys.argv[1]))
        except Exception:
            polynomial_fit_n_order(x, y, 2)

    else:
        polynomial_fit_n_order(x, y, 2)
