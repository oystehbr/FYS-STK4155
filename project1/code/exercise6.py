"""
Look into the main function - and uncomment the exercise you wanna run, 
we have restricted the images to be in some specific coordinates. Just go 
to the function read_terrain_data and set False in the if-statement to do the whole image.  
Enjoy!
"""


from imageio import imread
import exercise1
import exercise2
import exercise3
import exercise4
import exercise5
import matplotlib.pyplot as plt
import numpy as np
import helper


def read_terrain_data(filename):
    """
    Takes an image and break it down to coordinates and colors
    and returns it

    :param filename (str):
        the filename of the image

    :return tuple(np.ndarray, np.ndarray, np.ndarray, int, int):
        - x coordinates
        - y coordinates
        - colors
        - row length of the image
        - col length of the image
    """

    # Load the terrain
    terrain = imread(filename)

    row_length = np.shape(terrain)[0]
    col_length = np.shape(terrain)[1]

    # Set this to false, if you wanna look at the whole image
    if True:
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


def exercise1_test(filename, degree):
    """
    Read docstrings of main method in exercise 1
    """

    x_values, y_values, z_values, terrain, row_length, col_length = read_terrain_data(
        filename)
    exercise1.main(x_values, y_values, z_values, degree=degree)


def exercise2_test(filename, max_degree):
    """
    Read docstrings of main method in exercise 2
    """

    x_values, y_values, z_values, terrain, row_length, col_length = read_terrain_data(
        filename)
    exercise2.main(x_values, y_values, z_values, max_degree=max_degree)


def exercise3_test(filename, degree):
    """
    Read docstrings of main method in exercise 3
    """

    x_values, y_values, z_values, terrain, row_length, col_length = read_terrain_data(
        filename)
    exercise3.main(x_values, y_values, z_values, degree)


def exercise4_test(filename, max_degree, degree):
    """
    Read docstrings of main method in exercise 4
    """

    x_values, y_values, z_values, terrain, row_length, col_length = read_terrain_data(
        filename)
    exercise4.main(x_values, y_values, z_values, max_degree, degree)


def exercise5_test(filename, max_degree, degree):
    """
    Read docstrings of main method in exercise 5
    """

    x_values, y_values, z_values, terrain, row_length, col_length = read_terrain_data(
        filename)
    exercise5.main(x_values, y_values, z_values, max_degree, degree)


def terrain_prediction(filename, degree=1, method='OLS', lmbda=10):
    """
    Will show how the regression prediction "looks" with 
    regard to a image/terrain, with the given filename

    :param filename (str):
        the filename of the image
    :param degree (int):
        the order of the polynomial that defines the design matrix
    :param regression_method (str):
        the preffered regression method: OLS, RIDGE or LASSO
    :param lmbda (float):
        parameter used by Ridge and Lasso regression (lambda)

    :return None:
    """

    x_values, y_values, z_values, terrain, row_length, col_length = read_terrain_data(
        filename)

    x_train, x_test, y_train, y_test, z_train, z_test = helper.train_test_split(
        x_values, y_values, z_values)

    _, _, beta = helper.predict_output(
        x_train=x_train, y_train=y_train, z_train=z_train,
        x_test=x_test, y_test=y_test,
        degree=degree, regression_method=method,
        lmbda=lmbda
    )

    X = helper.create_design_matrix(x_values, y_values, degree)
    X_scaled = X - np.mean(X, axis=0)
    z_pred_all = X_scaled @ beta + np.mean(z_train, axis=0)

    # Transfer back to matrix
    terrain_pred = np.zeros((row_length, col_length))
    counter = 0
    for i in range(row_length):
        for j in range(col_length):
            terrain_pred[i][j] = z_pred_all[counter]
            counter += 1

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle(
        'The actual terrain vs. predicted', fontsize=20)

    # Line plots
    ax1.set_title('Terrain')
    ax1.imshow(terrain, cmap='gray')

    ax2.set_title(f'Predicted with: {method} ')
    ax2.imshow(terrain_pred, cmap='gray')

    plt.tight_layout()
    # Make space for title
    plt.subplots_adjust(top=0.85)
    plt.show()


def main():
    """
    In the function: read_terrain_data, you can set an if-statement
    to False for running the whole terrain - for now, it's restricted 
    by some coordinates (for running time sake)
    """

    # TODO: get the files from figures map -> set constant for the two filer
    TERRAIN_1 = 'terrain/SRTM_data_Norway_1.tif'
    TERRAIN_2 = 'terrain/SRTM_data_Norway_2.tif'
    IMAGE_ = 'terrain/case_real_2.tif'

    # terrain_prediction(filename=TERRAIN_2,
    #                    degree=30, method='OLS')
    # terrain_prediction(filename=IMAGE_, degree=5)

    # Exercise 1
    # exercise1_test(filename=TERRAIN_2, degree=1)

    # Exercise 2
    # exercise2_test(filename=TERRAIN_2, max_degree=2)

    # Exercise 3, do not take the whole picture -> scale down
    # exercise3_test(filename=TERRAIN_2, degree=2)

    # Exercise 4
    # exercise4_test(filename=TERRAIN_2, max_degree=1, degree=3)

    # Exercise 5
    # exercise5_test(filename=TERRAIN_2, max_degree=3, degree=2)


if __name__ == "__main__":
    main()
