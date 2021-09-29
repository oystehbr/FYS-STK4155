import exercise2
import exercise1
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

    ###
    row_length = np.shape(terrain)[0]
    col_length = np.shape(terrain)[1]

    row_length = 20
    col_length = 10

    m = 5  # polynomial order
    terrain = terrain[-row_length:, -col_length:]

    x_array = np.linspace(0, 1, col_length)   # x moves sideways
    y_array = np.linspace(0, 1, row_length)

    y_values = np.repeat(y_array, col_length)
    x_values = np.tile(x_array, row_length)
    z_values = terrain.ravel()

    return x_values, y_values, z_values, terrain, row_length, col_length


# TODO:


def terrain_prediction(filename, degree=1):
    """
    # TODO: docs

    """

    x_values, y_values, z_values, terrain, row_length, col_length = read_terrain_data(
        filename)

    # TODO: Call the different main-methods
    # exercise2.main(x_values, y_values, z_values, max_degree=8)
    # return

    x_train, x_test, y_train, y_test, z_train, z_test = exercise1.train_test_split(
        x_values, y_values, z_values)

    print(2)
    # Train the model with the training data
    X_train = helper.create_design_matrix(x_train, y_train, degree)
    print(2.5)
    X_test = helper.create_design_matrix(x_test, y_test, degree)

    print(3)
    # Scale data before further use
    X_train_scale = np.mean(X_train, axis=0)
    X_train_scaled = X_train - X_train_scale

    print(4)
    # TODO: shall we scale with the above?
    X_test_scaled = X_test - X_train_scale
    z_train_scale = np.mean(z_train, axis=0)
    z_train_scaled = z_train - z_train_scale

    print(5)
    # Get the betas from OLS.
    betas_OLS = exercise1.get_betas_OLS(X_train_scaled, z_train_scaled)

    print(6)
    # Scale the data back to its original form
    X = helper.create_design_matrix(x_values, y_values, degree)

    print(6.5)
    z_pred = exercise1.z_predicted(X, betas_OLS) + z_train_scale

    print(7)
    # Transfer back to matrix
    terrain_pred = np.zeros((row_length, col_length))
    counter = 0
    for i in range(row_length):
        for j in range(col_length):
            terrain_pred[i][j] = z_pred[counter]
            counter += 1

    print(8)
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

    terrain_prediction(filename='SRTM_data_Norway_2.tif', degree=1)
    terrain_prediction(filename='SRTM_data_Norway_2.tif', degree=1)


if __name__ == "__main__":
    main()
