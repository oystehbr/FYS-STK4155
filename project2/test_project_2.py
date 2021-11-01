"""
Setup for testing the results of project 2, feel free to change the 
different variables. Read docstrings inside the functions
to know what they do
"""

import numpy as np
import gradient_descent
import helper


test_gradient_decent = False
"""START the testing of gradient_descent.py """
if test_gradient_decent:
    # Generating the data (Franke Function)
    n = 500
    noise = 0.4
    x_values, y_values, z_values = helper.generate_data(n, noise)

    # Setting some preffered values
    list_number_of_minibatches = [1, 10, 20, 40,
                                  100, 400]  # TODO: not functioning
    list_number_of_minibatches = [1, 400]  # TODO: not functioning
    number_of_epochs = 20
    degree = 1  # complexity of the model
    gamma = 0.1  # the momentum of the stochastic gradient decent

    "Set to true, stochastic gradient decent testing with OLS"
    run_main_OLS = False
    if run_main_OLS:
        gradient_descent.main_OLS(
            x_values=x_values, y_values=y_values, z_values=z_values,
            list_no_of_minibatches=list_number_of_minibatches,
            n_epochs=number_of_epochs,
            degree=degree, gamma=gamma
        )

    no_of_minibatches = 10
    "Set to true, stochastic gradient decent testing with RIDGE"
    run_main_RIDGE = True
    gradient_descent.main_RIDGE(
        x_values=x_values, y_values=y_values, z_values=z_values,
        no_of_minibatches=no_of_minibatches,
        n_epochs=number_of_epochs,
        degree=degree, gamma=gamma
    )


test_Neural_Network = False
""" START the testing """
# TODO: finish this
if test_Neural_Network:

    pass
