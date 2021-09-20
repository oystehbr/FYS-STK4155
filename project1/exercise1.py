import helper
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import random
import numpy as np
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def franke_function(x, y):
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


def generate_data(n, noise_multiplier=1):
    """ 
    Generates data
    :param n (int):
        number of x and y values
    :param noise_multiplier (float, int):
        scale the noise
    :return (np.ndarray):
        array of generated funciton values with noise
    """

    # TODO: vectorize
    data_array = np.zeros(n)
    x_array = np.zeros(n)
    y_array = np.zeros(n)
    for i in range(n):
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        eps = np.random.normal(0, 1)
        z = franke_function(x, y) + noise_multiplier * eps
        x_array[i] = x
        y_array[i] = y
        data_array[i] = z
    return x_array, y_array, data_array


def get_betas(x_values, y_values, z_values, degree=2):
    """
    TODO: docstrings


    """

    # Creating the design matrix for our x- and y-values
    X = helper.create_design_matrix(x_values, y_values, degree)

    X_T = np.matrix.transpose(X)
    beta = np.linalg.inv(X_T @ X) @ X_T @ z_values
    # LÆRER SA DENNE VAR BEST:
    # beta = np.linalg.pinv(X_T @ X) @ X_T @ z_values -> SVD
    return beta


def main():
    n = 10000
    x_values, y_values, z_values = generate_data(n, 0)

    betas = get_betas(x_values, y_values, z_values, 3)
    print(betas)


#  f(x, y) = Dersom(0 < x < 1 ∧ 0 < y < 1, 1.19606753 - -1.02899952x + -0.75610173y + 0.06874209x² + 0.88365586x*y - -0.38809648y²)


if __name__ == "__main__":
    main()


# TODO: slett denne
def plot_polynomial(x_true, y_true, z_true, beta):
    """
    # TODO: docstrings
    """
    fig = plt.figure()
    fig1 = plt.figure()
    ax1 = fig.gca(projection='3d')
    ax = fig.gca(projection='3d')
    # Make data.

    beta, X = get_betas(x_true, y_true, z_true)

    x_pent = np.linspace(0, 1, 100)
    y_pent = np.linspace(0, 1, 100)
    x_mesh, y_mesh = np.meshgrid(x_true, y_true)
    z_true = franke_function(x_mesh, y_mesh)
    z_mesh = find_z_approx(x_mesh, y_mesh, z_true)

    print("c")

    z_approx = X @ beta

    # Plot the surface.
    surf = ax.plot_surface(x_mesh, y_mesh, z_true, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    surf1 = ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

    # # makeapproximated plot:
    # for i in range(len(x)):

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax1.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax1.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('% .02f'))
    ax1.zaxis.set_major_formatter(FormatStrFormatter('% .02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig1.colorbar(surf1, shrink=0.5, aspect=5)
    print("helooo")
    plt.show()
