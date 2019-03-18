import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def signum(x):
    return 1 if x >= 0 else -1


def Hebb_rule(self, i, j, train):
    weight = 0
    for p in range(train.shape[0]):
        weight += train[p, i] * train[p, j]
    return weight


class Hopfield:
    def __init__(self, number_of_bits, print_format, synchronous=False, activation_function=np.vectorize(signum),
               symmetric_weights=True, zero_self_interaction=True, weights_update=Hebb_rule):
        self.number_of_bits = number_of_bits
        self.synchronous = synchronous
        self.activation_function = activation_function
        self.weights = np.zeros((self.number_of_bits, self.number_of_bits))
        self.bias = np.zeros((self.number_of_bits,))
        self.symmetric_weights = symmetric_weights
        self.zero_self_interaction = zero_self_interaction
        self.weights_update = weights_update
        self.memory_state = np.zeros(self.number_of_bits)
        self.print_format = print_format

    def train(self, dataset):
        train_set = dataset.reshape(dataset.shape[0], -1)
        for i in range(self.number_of_bits):
            for j in range(self.number_of_bits):
                if self.zero_self_interaction and i == j:
                    continue
                elif self.symmetric_weights and j < i:
                    self.weights[i, j] = self.weights[j, i]
                else:
                    self.weights[i, j] = self.weights_update(self, i, j, train_set)
        self.weights /= self.number_of_bits
  
    def update(self, state=None):
        if state is None:
            state = self.memory_state
        else:
            state = state.flatten()
        if self.synchronous:
            return self.synchronous_update(state=state)
        else:
            return self.asynchronous_update(state=state)

    def synchronous_update(self, state):
        new_state = (self.weights @ state.reshape((self.number_of_bits, 1)))
        new_state += self.bias.reshape((self.number_of_bits,1))
        self.memory_state = self.activation_function(new_state)
        return self.memory_state.reshape(self.print_format)
  
    def asynchronous_update(self, state):
        new_state = state
        for i in range(self.number_of_bits):
            new_state[i] = self.activation_function(self.weights[i, :] @ new_state + self.bias[i])
        self.memory_state = new_state
        return new_state.reshape(self.print_format)
