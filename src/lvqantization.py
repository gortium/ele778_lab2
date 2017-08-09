import logging
import functools
import os
import sys
import yaml
import numpy as np


class LVQ:

    def __init__(self, nb_represent, nb_classe, nb_input, nb_output):
        np.random.seed(1)

        # Init variables
        self.nb_represent = nb_represent
        self.nb_classe = nb_classe
        self.nb_input = nb_input
        self.nb_output = nb_output
        self.X = None
        self.W = None
        self.d = [None] * self.nb_represent
        self.old_W = [0]
   #     self.B = [None] * self.nb_layer
   #     self.A = [None] * self.nb_layer
  #      self.I = [None] * self.nb_layer
 #       self.E = [None] * self.nb_layer
#        self.dW = [None] * self.nb_layer
 #       self.nb_hidden = nb_hidden
 #       self.activation = activation

    def distance(self, x, w):
        # Propogate inputs though network

        # Saving input
        self.X = x
        self.W = w
        square = 0
        v = 0   # vector position in vector array
        vn = 0   # number position in vector array
        r = 0   # representative position in representative array
        rn = 0   # number position in representative array

        # (vector classe, vector number, representative classe, distance from representative)
        distance = np.zeros((10,10,10,3))
        
        
        for v_classe in x:
            for vector in x[v_classe]:
                if len(x[v_classe]) != 0:
                    for r_classe in w:
                        for represent in w[r_classe]:
                            for  number in vector:
                                square += (number - represent[vn])**2
                                vn += 1
                            vn = 0
                            distance[v_classe][v][r][rn] = square
                            square = 0
                            rn += 1
                        rn = 0
                        r += 1
                    r = 0
                v += 1
            v = 0

        return distance

    # back propagation (push or pull representative)
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

    def predict(self, x, w):
        yhat = self.distance(x, w)
        return self.max_in(yhat)









