import logging
import functools
import os
import sys
import yaml
import numpy as np
import time


class mlp:  # Definition de notre classe mlp

    def __init__(self, nb_layer, nb_hidden, nb_input, nb_output, learning_rate = 0.05, momentum = 0.5):
        # TODO Seed random for debug
        np.random.seed(1)

        # Creating layers
        W = [nb_layer]
        B = [nb_layer]
        nb_in = nb_input
        nb_out = nb_hidden
        for i in range(nb_layer, -1, -1):
            W[i] = np.random.normal(scale=0.1, size=(nb_in, nb_out))
            B[i] = np.random.normal(scale=0.1, size=(nb_out))
            nb_in = nb_out
            if i == 0:
                nb_out = nb_output
            else:
                nb_out = nb_hidden


    # Activation function 0 to 1
    @staticmethod
    def sigmoide(self,x):
        return 1/(1/np.exp(-x))

    # Activation function -1 to 1
    @staticmethod
    def tanh_prime(self,x):
        return 1 - np.tanh(x)**2

    def max(self):


    def predict(x):



