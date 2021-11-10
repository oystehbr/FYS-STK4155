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

    # TODO: delete the link
    LINK: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU
    """
    if not deriv:
        # max(0, z) + negative_slope * min(0, z)
        return np.where(z >= 0, z, negative_slope*z)

    else:
        return np.where(z >= 0, 1, negative_slope)


def soft_max(z, deriv: bool = False):
    """
    Applying the softmax-function to our input data
    and return the values

    :param z (np.ndarray, number):
        the input value(s)
    :param deriv (bool):
        - True if we want the derivative
        - False if we want the function value

    :return (float):
        the function value

    # TODO: Maybe wrong written, maybe sum over some axis???
    Link: https://compphysics.github.io/MachineLearning/doc/pub/week41/html/week41.html
    """

    if not deriv:
        return np.exp(z)/np.sum(np.exp(z))
    else:
        # Link: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

        return np.exp(z)/np.sum(np.exp(z)) * \
            (1 - np.exp(z)/np.sum(np.exp(z)))


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
        try:
            ret = np.exp(-z)/((1+np.exp(-z))**2)
        except Exception or Warning as e:
            # Refreshing the variables
            print(f'>> (sigmoid derivative) ERROR: {e}')
            print('maybe turn down the complexity')
            exit()
    else:
        return 1/(1 + np.exp(-z))
        try:
            ret = 1/(1 + np.exp(-z))
        except Exception or Warning as e:
            # Refreshing the variables
            print(f'>> (sigmoid function) ERROR {e}')
            print('maybe reduce the complexity')
            exit()

    return ret


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
            print('maybe turn down the complexity')
            exit()
    else:
        try:
            ret = 1/(1 + np.exp(-z))
            ret = np.where(ret >= 0.5, 1, 0)
        except Exception or Warning as e:
            # Refreshing the variables
            print(f'>> ERROR {e}')
            print('maybe reduce the complexity')
            exit()

    return ret


def sigmoid_copy(self, x, deriv=False):
    """
    Apply the sigmoid activation function to
    scalar, vectors or matrices

    :param x (float):
        input value

    :return (float):
        the function value
    """
    if deriv:
        try:
            ret = np.exp(-x)/((1+np.exp(-x))**2)
        except Exception or Warning as e:
            # Refreshing the variables
            print(f'>> ERROR: {e}')
            print('maybe turn down the complexity')
            exit()
    else:
        try:
            ret = 1/(1 + np.exp(-x))
        except Exception or Warning as e:
            # Refreshing the variables
            print(f'>> ERROR {e}')
            print('maybe reduce the complexity')
            exit()

    return ret
