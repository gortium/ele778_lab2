import logging
import functools
import os
import sys
import yaml
import copy
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

    # compute the distance between vectors and all 30 representative (30 by default)
    def distance(self, x, w):

        # Saving input
        self.X = x
        self.W = w
        square = 0
        v = 0   # vector position in vector array
        vn = 0   # vector number position in vector array
        r = 0   # representative position in representative array
        rn = 0   # representative number position in representative array

        # (vector classes, vector quantity by classe, representative classes, distance from representative)
        distance = np.zeros((10,10,10,3))
        
        # could substitute the variable += 1 by index and enumerate function in the "for loops"
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

    # "m" contains 10 lists (1 for each classe) wich contains 10 vectors distributed in their corresponding classes.
    # all vectors contain the distance between itself and the 30 (by default) representative
    def closest_in(self, m):

        # "result" contains the 10 vectors (number per batches) with 4 info
        # the vector class, it's position number,
        # the representative's classe that is the closest and its position number
        result = np.zeros((10,4))

        v_nb = 0 # to identify the 10 vectors inside the "result" array

        for idx_c, classe in enumerate(m): # classe of vectors
            for idx_v, vector_nb in enumerate(classe): # vector number
                if vector_nb.max() != 0: # vector not empty?
                    for idx_r, represent in enumerate(vector_nb):
                        if vector_nb[idx_r].min() == vector_nb.min(): # is the closest representative in this class?
                            # go throug the 3 representatives of each representatives classe
                            for idx_i, represent_nb in enumerate(represent):
                                if represent[idx_i] == vector_nb.min(): # which of the 3 is it?
                                    result[v_nb][0] = idx_c # the vector class
                                    result[v_nb][1] = idx_v # the vector position number
                                    result[v_nb][2] = idx_r # the representative's classe that is the closest
                                    result[v_nb][3] = idx_i # the representative's position number
                                    v_nb += 1
                                    break # the closest representative has been found
                            break # the closest representative has been found
                else:
                    break # all further vector are NULL

        return result

    # push or pull representative with (w_new = w_old + learning_rate * |vector - w_old|)
    def push_pull(self, vector, w, closest, learning_rate):


        new_w = copy.deepcopy(w) # copy all representative to keep the ones that don't change

        for idx_vc, info_vector in enumerate(closest):

            classe_v = int(info_vector[0])
            vector_nb = int(info_vector[1])
            classe_r = int(info_vector[2])
            represent_nb = int(info_vector[3])
            x = vector[classe_v][vector_nb] # this is the vector info to iterate in
            old_w = copy.deepcopy(w[classe_r][represent_nb]) # this is the representative info to iterate in

            if info_vector[0] == info_vector[2]: # vector is the same class as representative
                for idx_i, nb_w in enumerate(old_w):
                    new_w[classe_r][represent_nb] = nb_w + (learning_rate * abs(x[idx_i] - nb_w))
            else: # not the same class
                for idx_i, nb_w in enumerate(old_w):
                    new_w[classe_r][represent_nb] = nb_w - (learning_rate * abs(x[idx_i] - nb_w))

        return new_w

    def predict(self, x, w):
        yhat = self.distance(x, w)
        return self.max_in(yhat)









