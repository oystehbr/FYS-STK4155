import exercise5
import exercise4
import exercise2
import exercise1
import exercise3
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy.random import normal, uniform
import helper


def read_terrain_data(filename):
    """
    #TODO:docs
    """
    # Load the terrain
    terrain = imread(filename)

    # print(terrain.shape)
    ###
    row_length = np.shape(terrain)[0]
    col_length = np.shape(terrain)[1]

    row_length = 250
    col_length = 300

    terrain = terrain[250:500, 1000:1300]

    x_array = np.linspace(0, 1, col_length)   # x moves sideways
    y_array = np.linspace(0, 1, row_length)

    y_values = np.repeat(y_array, col_length)
    x_values = np.tile(x_array, row_length)
    z_values = terrain.ravel()
    # print(z_values)

    return x_values, y_values, z_values, terrain, row_length, col_length


# TODO:

def exercise1_test(filename, degree):
    """
    # TODO: docstrings -> copy main() - docs
    """

    x_values, y_values, z_values, terrain, row_length, col_length = read_terrain_data(
        filename)
    exercise1.main(x_values, y_values, z_values, degree=degree)


def exercise2_test(filename, max_degree):
    """
    # TODO: docstrings -> copy main() - docs
    """

    x_values, y_values, z_values, terrain, row_length, col_length = read_terrain_data(
        filename)
    exercise2.main(x_values, y_values, z_values, max_degree=max_degree)


def exercise3_test(filename, degree):
    x_values, y_values, z_values, terrain, row_length, col_length = read_terrain_data(
        filename)
    exercise3.main(x_values, y_values, z_values, degree)


def exercise4_test(filename, max_degree, degree):
    x_values, y_values, z_values, terrain, row_length, col_length = read_terrain_data(
        filename)
    exercise4.main(x_values, y_values, z_values, max_degree, degree)


def exercise5_test(filename, max_degree, degree):
    x_values, y_values, z_values, terrain, row_length, col_length = read_terrain_data(
        filename)
    exercise5.main(x_values, y_values, z_values, max_degree, degree)


def terrain_prediction(filename, degree=1):
    """
    # TODO: docs

    """

    x_values, y_values, z_values, terrain, row_length, col_length = read_terrain_data(
        filename)

    x_train, x_test, y_train, y_test, z_train, z_test = helper.train_test_split(
        x_values, y_values, z_values)

    _, _, betas = helper.predict_output(
        x_train=x_train, y_train=y_train, z_train=z_train,
        x_test=x_test, y_test=y_test,
        degree=degree, regression_method='OLS',
    )

    X = helper.create_design_matrix(x_values, y_values, degree)
    X_scaled = X - np.mean(X, axis=0)
    z_pred_all = X_scaled @ betas + np.mean(z_train, axis=0)

    # Transfer back to matrix
    terrain_pred = np.zeros((row_length, col_length))
    counter = 0
    for i in range(row_length):
        for j in range(col_length):
            terrain_pred[i][j] = z_pred_all[counter]
            counter += 1

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle('Title of figure', fontsize=20)

    # Line plots
    ax1.set_title('Terrain')
    ax1.imshow(terrain, cmap='gray')

    ax2.set_title('Predicted Terrain')
    ax2.imshow(terrain_pred, cmap='gray')

    plt.tight_layout()
    # Make space for title
    plt.subplots_adjust(top=0.85)
    plt.show()


def main():

    # terrain_prediction(filename='SRTM_data_Norway_1.tif', degree=1)
    terrain_prediction(filename='case_real_2.tif', degree=25)
    # terrain_prediction(filename='pandas_real.tif', degree=50)

    # Exercise 1
    # exercise1_test(filename='SRTM_data_Norway_2.tif', degree=5)

    # # Exercise 2
    # exercise2_test(filename='SRTM_data_Norway_1.tif', max_degree=12)

    # # Exercise 3, do not take the whole picture -> scale down
    # exercise3_test(filename='SRTM_data_Norway_1.tif', degree=1)

    # Exercise 4
    # exercise4_test(filename='SRTM_data_Norway_1.tif', max_degree=5, degree=1)

    # Exercise 5
    # exercise5_test(filename='SRTM_data_Norway_1.tif', max_degree=5, degree=1)


if __name__ == "__main__":
    main()
