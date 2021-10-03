"""
Function for testing the different exercises,
please read docstrings if you want to know more 
about the functions we have been used
"""


import exercise1
import exercise2
import exercise3
import exercise4
import exercise5
import exercise6
import helper

# Set the preffered values
n = 100
noise = 0.2
degree = 8
test_size = 0.2
max_degree = 10
n_bootstrap = 100
k_folds = 5
x_values, y_values, z_values = helper.generate_data(n, noise)


# Reproducing exercise 1:
# Replace with True to test
if False:
    exercise1.main(
        x_values=x_values, y_values=y_values, z_values=z_values,
        degree=degree, test_size=test_size
    )

# Reproducing exercise 2:
# Replace with True to test
if False:
    exercise2.main(
        x_values=x_values, y_values=y_values, z_values=z_values,
        max_degree=max_degree, test_size=test_size,
        n_bootstrap=n_bootstrap
    )

# Reproducing exercise 3:
# Replace with True to test
if False:
    exercise3.main(
        x_values, y_values, z_values,
        degree=degree, k_folds=k_folds
    )

lmbda = 0.1
# Reproducing exercise 4:
# Replace with True to test
if False:
    exercise4.main(
        x_values=x_values, y_values=y_values, z_values=z_values,
        max_degree=max_degree, degree=degree,
        test_size=test_size, k_folds=k_folds,
        n_bootstrap=n_bootstrap, lmbda=lmbda
    )

# Reproducing exercise 5:
# Replace with True to test
if False:
    exercise5.main(
        x_values=x_values, y_values=y_values, z_values=z_values,
        max_degree=max_degree, degree=degree,
        test_size=test_size, k_folds=k_folds,
        n_bootstrap=n_bootstrap, lmbda=lmbda
    )

# Reproducing exercise 6:
# Replace with True to test
# TODO:
