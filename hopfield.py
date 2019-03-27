import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

@np.vectorize
def signum(x):
    return 1 if x >= 0 else -1

class Hopfield:
    def __init__(self, number_of_bits, print_format, synchronous=False, activation_function=signum,
               symmetric_weights=True, zero_self_interaction=True):
        self.number_of_bits = number_of_bits
        self.synchronous = synchronous
        self.activation_function = activation_function
        self.weights = np.zeros((self.number_of_bits, self.number_of_bits))
        self.bias = np.zeros((self.number_of_bits,))
        self.symmetric_weights = symmetric_weights
        self.zero_self_interaction = zero_self_interaction
        self.memory_state = np.zeros(self.number_of_bits)
        self.print_format = print_format

    def train(self, dataset):
        train_set = dataset.reshape(dataset.shape[0], -1)
        self.weights = (train_set.T @ train_set / self.number_of_bits).astype('float64')
        if self.zero_self_interaction:
            for i in range(self.number_of_bits):
                self.weights[i, i] = 0.
        if self.symmetric_weights:
            for i in range(self.number_of_bits):
                for j in range(i+1, self.number_of_bits):
                    self.weights[j, i] = self.weights[i, j]

  
    def update(self, state=None, print_form=True):
        if state is None:
            state = self.memory_state
        else:
            state = state.flatten()
        if self.synchronous:
            return self.synchronous_update(state=state, print_form=print_form)
        else:
            return self.asynchronous_update(state=state, print_form=print_form)

    def synchronous_update(self, state, print_form):
        new_state = (self.weights @ state.reshape((self.number_of_bits, 1)))
        new_state += self.bias.reshape((self.number_of_bits,1))
        self.memory_state = self.activation_function(new_state)
        if print_form:
            return self.memory_state.reshape(self.print_format)
        else:
            return self.memory_state

  
    def asynchronous_update(self, state, print_form):
        new_state = state
        ids = [i for i in range(self.number_of_bits)]
        random.shuffle(ids)
        for i in ids:
            new_state[i] = self.activation_function(self.weights[i, :] @ new_state + self.bias[i])
        self.memory_state = new_state
        if print_form:
            return self.memory_state.reshape(self.print_format)
        else:
            return self.memory_state
