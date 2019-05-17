import numpy as np
from random import shuffle
from utilities import sigmoid, linear
from denselayer import Dense_Layer
import loss

# ---------------------------------------------------------------------------------------------------------
# -----Shallow Network Class-------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------


class ShallowNetwork:
    '''
    Class representing a neural network with a single densly connected layer.
    It consits of the following fields:
    1. input_dim - dimension of the input vector,
    2. hidden_dim - number of units in the hidden layer,
    3. output_dim - dimension of the output,

    4. input_layer - zero_bias, identity, linear activation Dense_Layer,

    5. hidden_layer - regular Dense_Layer with full parameters and given activation function
        by activation_function_hidden property,
    5. output_layer - Dense_Layer with wights matrix, but zero bias and linear activation function,/
    It consists of the following methods described below ath their declaration:
    1. initialization,
    2. forward_pass,
    3. loss_function,
    4. gradient_descent_single_epoch
    5. train
    '''

    # --------------------------------------------------------------------------------
    def __init__(self, input_dim=1, hidden_dim=1, output_dim=1, activation_function_hidden=sigmoid):
        '''
        Initialization of the class:
        @params:
        1. input_dim - positive int, default 1,
        2. hidden_dim - positive int, default 1,
        3. output_dim - positive int, default 1,
        4. activation_function_hidden - callable, default sigmoid, passed to the hidden layer.
        '''
        # Dimensions of the layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.compiled = False
        self.layers = []
        # Input layer
        self.layers.append(Dense_Layer(
            input_dim=input_dim,
            neuron_number=input_dim,
            activation_function=linear,
            zero_bias=True,
            identity=True,
        ))
        # Hidden layer
        self.layers.append(Dense_Layer(
            input_dim=input_dim,
            neuron_number=self.hidden_dim,
            activation_function=activation_function_hidden))

        # Output layer
        self.layers.append(Dense_Layer(
            input_dim=hidden_dim,
            neuron_number=self.output_dim,
            activation_function=linear,
            zero_bias=False
        ))

    # --------------------------------------------------------------------------------
    # Forward pass

    def __call__(self, x):
        '''
        Passes the input vector through the network N(X)
        @params: x - numpy array, input vector
        @returns: numpy array
        '''
        response = x
        for layer in self.layers:
            response = layer(response)
        return response

    # --------------------------------------------------------------------------------
    # -----Learning methods-----------------------------------------------------------
    # --------------------------------------------------------------------------------

    # Updating the parameters via simple gradient descent
    def gradient_descent_step(self, X, Y, learning_rate, **kwargs):
        '''
        Performs a single epoch training on all samples and updates the weights of the network
        according to the simple online gradient descent backpropagation. 
        @params: 
        1. X - numpy array, samples,
        2. Y - numpy array, labels,
        3. learning_rate - float, learning rate.
        '''
        def update_layer(self, further_signal, incoming_signal, learning_rate):
            derivative = self.derivative(incoming_signal)
            
            delta_error = further_signal * derivative
            weight_change = -learning_rate * delta_error.T @ incoming_signal
            bias_change = -learning_rate * delta_error.T @ np.ones(shape=(incoming_signal.shape[0], 1))
            self.update_parameters(weight_change, bias_change)
            return delta_error @ self.weights        

        responses = [X]
        for layer in self.layers:
            responses.append(layer(responses[-1]))

        layers_to_train = [self.layers[len(self.layers) - i - 1]
                           for i in range(len(self.layers))]

        previous_delta = self.loss(X, Y, derivative=True, **kwargs)

        for id, layer in enumerate(layers_to_train):
            previous_delta = update_layer(layer,
                previous_delta, responses[len(responses) - id - 2], learning_rate)

    # Updating the parameters via Adagrad
    def adagrad_step(self, X, Y, learning_rate, **kwargs):
        '''
        Performs a single epoch training on all samples and updates the weights of the network
        according to the simple online gradient descent backpropagation. 
        @params: 
        1. X - numpy array, samples,
        2. Y - numpy array, labels,
        3. learning_rate - float, learning rate.
        '''
        def update_layer(self, further_signal, incoming_signal, learning_rate,
            weight_factor, bias_factor):
            derivative = self.derivative(incoming_signal)
            
            delta_error = further_signal * derivative
            weight_gradient = delta_error.T @ incoming_signal
            weight_mod = np.power(weight_factor + 1e-8, -0.5)
            weight_change = -learning_rate * weight_gradient * weight_mod
        

            bias_gradient = delta_error.T @ np.ones(shape=(incoming_signal.shape[0], 1))
            bias_mod = np.power(bias_factor + 1e-8, -0.5)
            bias_change = -learning_rate * bias_gradient * bias_mod
            
            self.update_parameters(weight_change, bias_change)
            return delta_error @ self.weights, weight_gradient, bias_gradient     

        responses = [X]
        for layer in self.layers:
            responses.append(layer(responses[-1]))

        layers_to_train = [self.layers[len(self.layers) - i - 1]
                           for i in range(len(self.layers))]

        previous_delta = self.loss(X, Y, derivative=True, **kwargs)

        for id, layer in enumerate(layers_to_train):
            previous_delta, weight_gradient, bias_gradient = update_layer(layer,
                previous_delta, responses[len(responses) - id - 2], learning_rate,
                weight_factor=self.weight_gradients[len(self.weight_gradients) - id - 1],
                bias_factor=self.bias_gradients[len(self.bias_gradients) - id - 1])
            self.weight_gradients[len(self.weight_gradients) - id - 1] += weight_gradient ** 2
            self.bias_gradients[len(self.bias_gradients) - id - 1] += bias_gradient ** 2
            

    # --------------------------------------------------------------------------------
    # Multiple epoch training
    def fit(self, X, Y, epochs, batch_size, learning_rate, verbose=True,
        message_frequency=1, **kwargs):
        '''
        Performs the given number of training epochs and prints the current loss function. 
        @params: 
        1. X - numpy array, set of input vectors,
        2. Y - numpy array, labels,
        3. epochs - positive integer, number of steps in the training
        4. verbose, boolean, wether to print training messages,
        5. learning_rate - float, learning rate,
        6. batch_size - non-negative int, number of samples taken into one batch.
        '''
        if not self.compiled:
            raise Exception('The model is not compiled!')
        if not isinstance(X, np.ndarray):
            raise Exception('X should be a numpy array.')
        if not isinstance(Y, np.ndarray):
            raise Exception('Y should be a numpy array.')
        if not isinstance(batch_size, int) or batch_size < -1 or batch_size == 0:
            raise Exception(
                'batch_size parameter should ba a non-negative integer or -1')
        if not isinstance(learning_rate, float):
            raise Exception(
                'learning_rate should be a floating point parameter')
        if isinstance(epochs, int) and epochs > 0:
            for epoch in range(epochs):
                if batch_size == -1:
                    batch_size = X.shape[0]
                batched_counter = 0
                while batched_counter < X.shape[0]:
                    if batched_counter + batch_size <= X.shape[0]:
                        self.training_step(X=X[batched_counter:batched_counter+batch_size, :],
                                           Y=Y[batched_counter:batched_counter+batch_size, :],
                                           learning_rate=learning_rate, **kwargs)
                    else:
                        self.training_step(X=X[batched_counter:, :],
                                           Y=Y[batched_counter:, :],
                                           learning_rate=learning_rate, **kwargs)
                    batched_counter += batch_size
                if verbose and epoch % message_frequency == 0:
                    print(
                        f'Epoch: {epoch+1} Loss function: {self.loss(samples=X, labels=Y, **kwargs)}')
            print(f'Final loss function: {self.loss(samples=X, labels=Y, **kwargs)}')
        else:
            raise Exception(f'Parameter epochs has to be a positive integer.')

    # Setting loss and optimizer
    def compile(self, loss_name, optimizer):
        '''
        Sets loss and optimizer for the training procedure. 
        @params: 
        1. loss_name - string, name of the loss, so far: mse, rmse, cross_entropy, chi_squared,
        2. optimizer - string, name of the optimizer, so far: GD (gradient descent) and Adagrad.
        '''
        if loss_name == 'rmse':
            self.loss_function_to_wrap = loss.rmse
        elif loss_name == 'mse':
            self.loss_function_to_wrap = loss.mse
        elif loss_name == 'cross_entropy':
            self.loss_function_to_wrap = loss.cross_entropy
        elif loss_name == 'chi_squared':
            self.loss_function_to_wrap = loss.chi_squared
        else:
            raise Exception(
                'Only RMSE, MSE, Chi-squared, and Crossentropy are supported as loss functions.')
        if optimizer == 'GD':
            self.training_step = self.gradient_descent_step
        elif optimizer == 'Adagrad':
            self.training_step = self.adagrad_step
            self.weight_gradients = []
            self.bias_gradients = []
            for layer in self.layers:
                self.weight_gradients.append(np.ones(shape=layer.weights.shape, dtype='float64'))
                self.bias_gradients.append(np.ones(shape=layer.bias.shape, dtype='float64'))
        else:
            raise Exception(
                'Only SGD and Adagrad are accepted as the optimizers.')
        self.compiled = True

    def loss(self, samples, labels, derivative=False, **kwargs):
        '''
        Wrapper for the supplied loss function.
        @params: 
        1. samples - numpy array, set of input vectors,
        2. labels - numpy array, in case of supervised,
        3. derivative - boolean, for training purposes.
        '''
        return self.loss_function_to_wrap(self, samples, labels, derivative, **kwargs)
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
