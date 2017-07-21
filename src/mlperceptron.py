import logging
import functools
import os
import sys
import yaml
import numpy as np


class MLP:

    def __init__(self, nb_layer, nb_input, nb_hidden, nb_output, activation):
        np.random.seed(1)

        # Init variables
        self.nb_layer = nb_layer
        self.X = None
        self.W = [None] * self.nb_layer
        self.B = [None] * self.nb_layer
        self.A = [None] * self.nb_layer
        self.I = [None] * self.nb_layer
        self.E = [None] * self.nb_layer
        self.dW = [None] * self.nb_layer
        self.old_dW = [0] * self.nb_layer
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.nb_hidden = nb_hidden
        self.activation = activation

        # Creating layers
        layer_in = self.nb_input
        layer_out = self.nb_hidden
        for layer in range(nb_layer):
            self.W[layer] = np.random.normal(scale=0.1, size=(layer_in, layer_out))
            self.B[layer] = np.random.normal(scale=0.1, size=layer_out)

            # Preparing variable for next layer
            layer_in = layer_out
            if layer == nb_layer - 2:
                layer_out = self.nb_output
            else:
                layer_out = self.nb_hidden

    # Activation function 0 to 1
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_prime(self, x):
        return np.exp(-x)/((1+np.exp(-x))**2)

    # Activation function -1 to 1
    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1 - np.tanh(x)**2

    def feed_forward(self, x):
        # Propogate inputs though network

        # Saving input
        self.X = x

        # For each layer,
        for layer in range(self.nb_layer):

            # Calculate activation
            if layer == 0:
                self.I[layer] = np.dot(x, self.W[layer]) + self.B[layer]
            else:
                self.I[layer] = np.dot(self.A[layer - 1], self.W[layer]) + self.B[layer]

            if self.activation == "sigmoid":
                self.A[layer] = self.sigmoid(self.I[layer])
            elif self.activation == "tanh":
                self.A[layer] = self.tanh(self.I[layer])

        return self.A[-1]

    # back propagation using batch gradient descent
    def backprop(self, yhat, Y, learning_rate, momentum):

        for layer in reversed(range(self.nb_layer)):

            # Calculating error
            if layer == self.nb_layer - 1:
                self.E[layer] = np.multiply((Y - yhat), self.sigmoid_prime(self.I[layer]))
            else:
                self.E[layer] = np.dot(self.E[layer + 1], self.W[layer + 1].T) * self.sigmoid_prime(self.I[layer])

            # Calculating dW
            if layer == 0:
                self.dW[layer] = np.dot(self.X.T, self.E[layer]) * learning_rate + self.old_dW[layer] * momentum
            else:
                self.dW[layer] = np.dot(self.A[layer - 1].T, self.E[layer]) * learning_rate + self.old_dW[layer] * momentum

        # Adjusting weight by dW
        for layer in range(self.nb_layer):
            self.W[layer] += self.dW[layer]

        self.old_dW = self.dW


    def max_in(self, m):
        result = np.zeros_like(m)
        result[np.arange(len(m)), m.argmax(1)] = 1
        return result

    def predict(self, x):
        yhat = self.feed_forward(x)
        return self.max_in(yhat)









