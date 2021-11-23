import numpy as np
np.seterr(all='warn')
np.seterr(over='raise')


def RELU(z, deriv: bool = False):
    """
    Applying the Rectified Linear Unit to our input data
    and return the values

    :param z (np.ndarray, number):
        the values to run through the RELU-function
    :param deriv (bool):
        - True if we want the derivative
        - False if we want the function value

    :return (float):
        the function value
    """

    if not deriv:
        # the relu function
        return np.where(z >= 0, z, 0)
    else:

        # the derivative of the relu function
        return np.where(z >= 0, 1, 0)


def Leaky_RELU(z, negative_slope: float = 0.01, deriv: bool = False):
    """
    Applying the Leaky Rectified Linear Unit to our input data
    and return the values

    :param z (np.ndarray, number):
        the input value(s)
    :param negative_slope (number), default=0.01:
        parameter used in the Leaky RELU function
    :param deriv (bool):
        - True if we want the derivative
        - False if we want the function value

    :return (float):
        the function value

    LINK: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU
    """
    if not deriv:
        # max(0, z) + negative_slope * min(0, z)
        return np.where(z >= 0, z, negative_slope*z)

    else:
        return np.where(z >= 0, 1, negative_slope)


def sigmoid(z, deriv=False):
    """
    Apply the sigmoid activation function to
    scalar, vectors or matrices

    :param z (np.ndarray):
        input value
    :param deriv (bool):
        - True if we want the derivative
        - False if we want the function value

    :return (float):
        the function value
    """
    if deriv:
        return np.exp(-z)/((1+np.exp(-z))**2)

    else:
        return 1/(1 + np.exp(-z))


def sigmoid_classification(z, deriv=False):
    """
    Apply the sigmoid activation function to
    scalar, vectors or matrices

    :param z (np.ndarray):
        input value
    :param deriv (bool):
        - True if we want the derivative
        - False if we want the function value

    :return (float):
        the function value
    """
    if deriv:
        try:
            ret = np.exp(-z)/((1+np.exp(-z))**2)
        except Exception or Warning as e:
            # Refreshing the variables
            print(f'>> ERROR: {e}')
            print('maybe turn down the complexity/ change eta')
            exit()
    else:
        try:
            ret = 1/(1 + np.exp(-z))
            ret = np.where(ret >= 0.5, 1, 0)
        except Exception or Warning as e:
            # Refreshing the variables
            print(f'>> ERROR {e}')
            print('maybe reduce the complexity/ change eta')
            exit()

    return ret
