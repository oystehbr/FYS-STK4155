from numpy import np


def RELU(z, deriv:bool=False):
    """
    Applying the Rectified Linear Unit
    to our input data
    """

    if not deriv:
        # the relu function
        return max(0, z)
    else:
        # the derivative of the relu function
        return np.where(z>0, 1, 0)



def Leaky_RELU(z, negative_slope:float=0.01, deriv:bool=False):
    """
    Applying the Leaky Rectified Linear Unit 
    to our input data
    LINK: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html#torch.nn.LeakyReLU
    """
    if not deriv:
        return max(0, z) + negative_slope * min(0, z)

    else:
        return np.where(z>0, 1, negative_slope)




def soft_max(z, deriv:bool=False):
    """
    # TODO: Maybe wrong written, maybe sum over some axis???


    Link: https://compphysics.github.io/MachineLearning/doc/pub/week41/html/week41.html
    """


    if not deriv:
        return np.exp(z)/np.sum(np.exp(z))




def sigmoid(z, deriv=False):
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
                ret = np.exp(-z)/((1+np.exp(-z))**2)
            except Exception or Warning as e:
                # Refreshing the variables
                print(f'>> ERROR: {e}')
                print('maybe turn down the complexity')
                exit()
        else:
            try:
                ret = 1/(1 + np.exp(-z))
            except Exception or Warning as e:
                # Refreshing the variables
                print(f'>> ERROR {e}')
                print('maybe reduce the complexity')
                exit()

        return ret