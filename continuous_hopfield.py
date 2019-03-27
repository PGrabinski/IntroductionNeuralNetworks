import numpy as np
import hopfield

class ContinuousHopifield(hopfield.Hopfield):
  def __init__(self, number_of_bits, print_format, synchronous_update=True):
    super().__init__(number_of_bits, print_format=print_format, synchronous=synchronous_update,
                     activation_function=lambda x: np.tanh(0.5 * x),
               symmetric_weights=True, zero_self_interaction=True)
    self.weights = 2*np.random.random((number_of_bits, number_of_bits)) -1
    
  def predict(self, state=None):
    pattern = state
    if pattern is None:
      pattern = self.memory_state
    if self.synchronous:
      self.memory_state = self.activation_function(self.weights @ pattern)
    else:
      new_state = np.copy(pattern)
      ids = np.arange(self.number_of_bits)
      np.random.shuffle(ids)
      for i in ids:
        temp = np.copy(new_state)
        new_state[i] = self.activation_function(self.weights[i, :] @ new_state)
      self.memory_state = new_state
    return self.memory_state
  
  def loss(self, dataset):
    train_set = dataset.reshape(dataset.shape[0], -1)
    response = np.apply_along_axis(self.predict, 1, train_set)
    return -0.5*((train_set+1-1e-31)*np.log(0.5*(response+1 +1e-31))
                 + (-train_set+1-1e-31)*np.log(0.5*(-response+1 +1e-31))).sum()
  
  def train(self, dataset, epochs=None, learning_rate=0.1, tolerance=None, verbose=False):
    train_set = np.copy(dataset.reshape(dataset.shape[0], -1))
    super().train(train_set)
    if not tolerance is None and epochs is None:
      prev_loss = self.loss(train_set) + 2 * tolerance
      # Trains as long as loss goes down by some given parameter
      counter = 1
      while self.loss(train_set) > tolerance:
        weight_change = np.apply_along_axis(self.predict, 1, train_set)
        weight_change = train_set + weight_change
        weight_change = -.5*train_set.T @ weight_change
        prev_loss = self.loss(train_set)
        if verbose:
          print(f'Epoch: {counter} Loss: {prev_loss}')
          counter += 1
        self.weights += -learning_rate * weight_change
        
    elif tolerance is None and not epochs is None:
      # Trains for a given number of epochs
      for i in range(epochs):
        weight_change = np.apply_along_axis(self.predict, 1, train_set)
        weight_change = (train_set + weight_change)
        weight_change = -0.5*train_set.T @ weight_change
        self.weights += -learning_rate * weight_change
        if verbose:
          print(f'Epoch: {i} Loss: {self.loss(train_set)}')
    else:
      raise Exception('Choose either precision or epoch training!')

#       if train_scipy:
#           def loss_to_minimize(weights, set_train, network):
# #         print(weights)
# #         print(network.loss(set_train))
#         network.weights = weights.reshape((self.number_of_bits,self.number_of_bits))
#         loss = network.loss(set_train)
# #         print(loss)
#         return loss
#       sol = minimize(fun=loss_to_minimize, x0=self.weights.flatten(), args=(train_set, self),
#                      method = 'l-bfgs-b', bounds=tuple([(-1,1) for x in range(self.number_of_bits**2)]),
#                     options={'gtol':1e-40, 'ftol':10.0})
#       print(sol)
#       self.weights = sol.x.reshape((self.number_of_bits, self.number_of_bits))
    
#     el