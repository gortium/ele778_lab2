import logging
import functools
import os
import sys
import yaml
import numpy as np
import time


class MLP:  # Definition de notre classe mlp

    def __init__(self, nb_layer, nb_hidden, nb_input, nb_output, learning_rate = 0.05, momentum = 0.5):
        # TODO Seed random for debug
        np.random.seed(1)

        # Init variables
        self.nb_layer_ = nb_layer
        self.W_ = [self.nb_layer_]      # ex: nb_layer_ = 2 --> W_ = [ , ]
        self.B_ = [self.nb_layer_]
        self.nb_input_ = nb_input
        self.nb_output_ = nb_output
        self.nb_hidden_ = nb_hidden

        # Creating layers
        layer_in = self.nb_input_
        layer_out = self.nb_hidden_
        for i in range(nb_layer-1):
            self.W[i] = np.random.normal(scale=0.1, size=(layer_in, layer_out))
            self.B[i] = np.random.normal(scale=0.1, size=(layer_out))

            # Preparing variable for next layer
            layer_in = layer_out
            if i == 0:
                layer_out = self.nb_output_
            else:
                layer_out = self.nb_hidden_

    # Activation function 0 to 1
    @staticmethod
    def sigmoide(self, x):
        return 1/(1/np.exp(-x))\

    @staticmethod
    def sigmoide_prime(self, x):
        return np.ex(-x)/((1+np.exp(-x))**2)

    # Activation function -1 to 1
    @staticmethod
    def tanh(self,x):
        return np.tanh(x)

    @staticmethod
    def tanh_prime(self,x):
        return 1 - np.tanh(x)**2

# #TODO
#     def max(self):
#
# #TODO
#     def predict(x):