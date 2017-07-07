import logging
import functools
import os
import sys
import yaml
import numpy as np
import time


class MLP:

    def __init__(self, nb_layer, nb_hidden, nb_input, nb_output, learning_rate = 0.05, momentum = 0.5):
        # TODO Seed random for debug
        np.random.seed(1)

        # Init variables
        self.nb_layer = nb_layer
        self.W = [self.nb_layer]      # ex: nb_layer_ = 2 --> W_ = [ , ]
        self.B = [self.nb_layer]
        self.A = [self.nb_layer]
        self.I = [self.nb_layer]
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.nb_hidden = nb_hidden

        # Creating layers
        layer_in = self.nb_input
        layer_out = self.nb_hidden
        for i in range(nb_layer-1):
            self.W[i] = np.random.normal(scale=0.1, size=(layer_in, layer_out))
            self.B[i] = np.random.normal(scale=0.1, size=layer_out)

            # Preparing variable for next layer
            layer_in = layer_out
            if i == 0:
                layer_out = self.nb_output
            else:
                layer_out = self.nb_hidden

        self.W = np.array(self.W)
        self.B = np.array(self.B)

    def feed_forward(self, batch):
        # Propogate inputs though network
        for layer_nb in range(self.nb_layer):
            self.I[layer_nb] = np.dot(batch, self.W[layer_nb])
            self.A[layer_nb] = self.sigmoid(self.I[layer_nb])

        yHat = self.sigmoid(self.I[-1])

        return yHat

    # Activation function 0 to 1
    def sigmoid(self, x):
        return 1/(1/np.exp(-x))

    def sigmoid_prime(self, x):
        return np.exp(-x)/((1+np.exp(-x))**2)

    # Activation function -1 to 1
    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1 - np.tanh(x)**2

    def cost_function(self, batch, y):
        # Compute cost for given X,y, use weights already stored in class.
        yHat = self.feed_forward(batch)
        S = 0.5 * sum((y - yHat) ** 2)
        return S

    def costFunction_prime(self, batch, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.feed_forward(batch)

        yHat = self.yHat
        dJdW = []
        last_delta = None
        for layer_nb in reversed(range(self.nb_layer)):
            # Last layer (first to be calculated)
            if layer_nb == self.nb_layer[-1]:
                delta = np.multiply(-(y - yHat), self.sigmoid_prime(self.I[layer_nb]))
                dJdW[layer_nb] = np.dot(self.A[layer_nb - 1].T, delta)
                last_delta = delta

            # First layer (last to be calculated)
            elif layer_nb == self.nb_layer[0]:
                delta = np.dot(last_delta, self.W[layer_nb - 1].T) * self.sigmoid_prime(self.I[layer_nb - 1])
                dJdW[layer_nb] = np.dot(batch.T, delta)

            # Other hidden layer
            else:
                delta = np.dot(last_delta, self.W[layer_nb - 1].T) * self.sigmoid_prime(self.I[layer_nb - 1])
                dJdW[layer_nb] = np.dot(self.A[layer_nb - 1].T, delta)
                last_delta = delta

        return dJdW1, dJdW2

    def compute_gradients(self, batch, y):
        dJdW1, dJdW2 = self.costFunction_prime(batch, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    # Helper Functions for interacting with other classes:
    def get_params(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def set_params(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.nb_hidden * self.nb_input
        self.W1 = np.reshape(params[W1_start:W1_end], (self.nb_input, self.nb_hidden))
        W2_end = W1_end + self.nb_hidden * self.nb_output
        self.W2 = np.reshape(params[W1_end:W2_end], (self.nb_hidden, self.nb_output))

# #TODO
#     def max(self):
#
# #TODO
#     def predict(x):



