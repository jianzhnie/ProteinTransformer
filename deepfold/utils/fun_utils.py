import numpy as np


def sigmoid(Z):
    """Implements the sigmoid activation in bumpy.

    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(z), same shape as Z
    """
    A = 1 / (1 + (np.exp((-Z))))

    return A
