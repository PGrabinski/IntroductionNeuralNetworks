import numpy as np
# ---------------------------------------------------------------------------------------------------------
# -----Activation Functions--------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------
# Sigmoid activation function
# ---------------------------------------------------------------------------------------------------------


def sigmoid(x, derivative):
    '''
    Sigmoid activation function.
    @params:
    1. x - float, the argument,
    2. deriv - non-negative integer, 0 for the function, 1 for its derivative
    @returns: float
    '''
    x = np.maximum(-5, x)
    x = np.minimum(5, x)
    temp_sig = np.exp(-x, dtype="float64")
    temp_sig = 1 / (temp_sig + 1)
    if derivative == 0:
        return temp_sig
    elif derivative == 1:
        return temp_sig * (1 - temp_sig)
    else:
        raise Exception(
            f'Parameter derivative should be either 0 or 1 but derivative = {derivative}')
# ---------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------
# Linear activation function
# ---------------------------------------------------------------------------------------------------------


def linear(x, derivative):
    '''
    Linear activation function and its derivatives.
    @params:
    1. x - float, the argument,
    2. deriv - non-negative integer, 0 for the function, 1 for its derivative
    @returns: float
    '''
    if derivative == 0:
        return x
    elif derivative == 1:
        return np.ones(shape=x.shape, dtype='float64')
    elif derivative > 1:
        return np.zeros(shape=x.shape, dtype='float64')
    else:
        raise Exception(
            f'Parameter derivative should be non-negative integer, but derivative = {derivative}')
# ---------------------------------------------------------------------------------------------------------
