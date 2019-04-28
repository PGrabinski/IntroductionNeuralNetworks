import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, loss_function, input_dimension):
        # Checking input dimension
        if isinstance(input_dimension, int) and input_dimension > 0:
            self.input_dimension = input_dimension
        else:
            raise Exception(
            'The input_dimension parameter should be a positive integer.')

        # Setting the loss function and activation function
        if loss_function == 'MSE':
            self.loss_function = loss_function
            self.activation = lambda x: np.power(1+np.exp(-x), -1)
        elif loss_function == 'Cross_entropy':
            self.loss_function = loss_function
            self.activation = lambda x: np.power(1+np.exp(-x), -1)
        else:
            raise Exception(
                'Only MSE and Cross_entropy are supported as loss_function functions')

        # Initialize parameters
        self.weights = np.random.random(size=(1, self.input_dimension)) * 1 - 0.5
        self.bias = np.random.random() * 1 - 0.5

    def __call__(self, X):
        if X.shape[1] == self.input_dimension:
            return self.activation(self.weights @ X.T + self.bias)
        else:
            raise Exception('The X array has invalid dimensions.')
      
    def loss(self, X, Y):
        if self.loss_function == 'MSE':
            loss = self(X)-Y.T
            return 0.5*np.sum(loss**2)
        elif self.loss_function == 'Cross_entropy':
            # Numerically 0 * log(0) is not realy 0
            epsilon = 1e-8 

            response = self(X)
            loss = (Y.T * np.log(response + epsilon)
                    + (1 - Y.T) * np.log(1 - response + epsilon))
            return -loss.sum()
    
    def weights_update(self, X, Y, learning_rate, batch_size):
        if self.loss_function == 'MSE':
            self.weights_update_MSE(X, Y, learning_rate, batch_size)
        elif self.loss_function == 'Cross_entropy':
            self.weights_update_CE(X, Y, learning_rate, batch_size)
    
    def weights_update_MSE(self, X, Y, learning_rate, batch_size):
        response = self(X)
        deriv = (response - Y.T) *  (1 - response) * response 
        self.weights -= learning_rate * deriv @ X / batch_size
        self.bias -= learning_rate * deriv.mean()
    
    def weights_update_CE(self, X, Y, learning_rate, batch_size):
        self.weights -= -learning_rate * ((Y.T - self(X)) @ X) / batch_size
        self.bias -= -learning_rate * (Y.T - self(X)).mean()

    def fit(self, X, Y, epochs, learning_rate, verbose=True, batch_size=1, message_frequency=1):
        # Online training is the case n=1 of the batch learning
        if not isinstance(epochs, int) or not epochs > 0:
            raise Exception('The epochs parameter should be a positive integer.')
        if not isinstance(batch_size, int) or not batch_size > 0:
            raise Exception('The batch_size parameter should be a positive integer.')

        #Recording the loss function
        history = []

        # Epoch loop
        for epoch in range(epochs):

            # Counter of the samples that were already used in the training
            batched = 0

            # batch loop - could be transformed into for loop, but this seems more concise
            while batched < Y.shape[0]:
                if batched + batch_size <= Y.shape[0]:
                    self.weights_update(X[batched:batched+batch_size],
                                    Y[batched:batched+batch_size],
                                    learning_rate, batch_size)
                else:
                    self.weights_update(X[batched:], Y[batched:],
                        learning_rate, batch_size)
                # Incrementing the counter
                batched += batch_size

            # Computing the loss, recording it, and printing if verbose mode is on
            epoch_loss = self.loss(X, Y)
            history.append(epoch_loss)
            if verbose and epoch % message_frequency == 0:
                print(f'Epoch: {epoch+1} Loss: {epoch_loss}')
                
        return history
