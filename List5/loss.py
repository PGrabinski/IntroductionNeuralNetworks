import numpy as np
'''
Available loss functions with appropiate derivatives
'''


# Root mean square error loss function
def rmse(self, samples, labels, derivative=False):
    '''
    Loss = sqrt(1/N sum( (samples - labels) ^ 2 ))
    @params: 
    1. samples - numpy array, set of input vectors,
    2. labels - numpy array, set of target vectors,
    3. derivative - boolean, if derivative of the loss.
    '''
    # Regular loss
    if not derivative:
        loss = ((self(samples) - labels) ** 2).sum()
        n_inv = samples.shape[0] ** -1
        loss *= n_inv
        loss = np.sqrt(loss)
        return loss
    # Derivative of the loss
    else:
        change = self(samples) - labels
        n_inv = samples.shape[0] ** -1
        derivative = np.power((change ** 2) * n_inv, -0.5) * n_inv * change
        return derivative

# Mean square error loss function
def mse(self, samples, labels, derivative=False):
    '''
    Loss = 1/N sum( (samples - labels) ^ 2 )
    @params: 
    1. samples - numpy array, set of input vectors,
    2. labels - numpy array, set of target vectors,
    3. derivative - boolean, if derivative of the loss.
    '''
    # Regular loss
    if not derivative:
        loss = 0.5*((self(samples) - labels) ** 2).mean()
        return loss
    # Derivative of the loss
    else:
        derivative = (self(samples) - labels)
        return derivative

# Cross entropy loss function
def cross_entropy(self, samples, labels, derivative=False):
    '''
    Loss = - sum( labels * log(samples) )
    @params: 
    1. samples - numpy array, set of input vectors,
    2. labels - numpy array, set of target vectors,
    3. derivative - boolean, if derivative of the loss.
    '''
    # Regular loss
    if not derivative:
        loss = - labels * np.log(samples + 1e-8)
        return loss.sum()
    # Derivative of the loss
    else:
        derivative =  -labels * np.power(samples + 1e-8, -1)
        return derivative

# Chi-squared loss function
def chi_squared(self, samples, labels, derivative=False, **kwargs):
    '''
    Loss = sum( (samples - labels)^2 / labels )
    @params: 
    1. samples - numpy array, set of input vectors,
    2. labels - numpy array, set of target vectors,
    3. derivative - boolean, if derivative of the loss.
    '''
    deltas = kwargs['DY']
    # Regular loss
    if not derivative:
        loss = (self(samples) - labels / deltas) ** 2
        return loss.sum()
    # Derivative of the loss
    else:
        derivative =  2 * (self(samples) - labels) / deltas**2
        return derivative