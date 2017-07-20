import logging
import os
import sys
import yaml
import functools
import numpy as np
#import mlp
import data_manager
import pickle
import random


class Trainer:
    def __init__(self, *existing_mlp):  # Constructeur
        # Hyperparameters
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.batch_size = 50
        self.n_epoch = 100
        self.activation_function = "sigmoid"

        # Supervising matrix
        D = np.zeros((10, 10), float)
        np.fill_diagonal(D, 1)

        self.data_manager = data_manager.DataManager()
        self.data_tree = None
        self.batchs = [self.n_epoch]

        # if existing_mlp:
        #     self.mlp = existing_mlp
        # else:
        #     self.mlp = mlp.MLP()

    # Create all the epoch baths, need to know: combined, static or dynamic data ?
    def create_batch(self, witch_filter):
        # Load data
        if witch_filter == "combined":
            self.data_manager.fetch_data(self.data_manager.trees["combined"],
                                         os.path.join(self.data_manager.paths["abs_filtered_data_path"], "combined"))
            self.data_tree = self.data_manager.trees["combined"]
        elif witch_filter == "static":
            self.data_manager.fetch_data(self.data_manager.trees["static"],
                                         os.path.join(self.data_manager.paths["abs_filtered_data_path"], "static"))
            self.data_tree = self.data_manager.trees["static"]
        elif witch_filter == "dynamic":
            self.data_manager.fetch_data(self.data_manager.trees["dynamic"],
                                         os.path.join(self.data_manager.paths["abs_filtered_data_path"], "dynamic"))
            self.data_tree = self.data_manager.trees["dynamic"]
        else:
            return False

        if self.activation_function == "sigmoid":
            self.data_manager.normalize_tree(self.data_tree, 0, 1)
        elif self.activation_function == "tanh":
            self.data_manager.normalize_tree(self.data_tree, -1, 1)

        for batch in self.batchs:
            batch = np.array()
            for data in batch:
                self.data_manager.find_random_and(self.data_tree["test"], self.data_manager.np_concatonate, batch)

    # input data, transpose, layers, biases, mlp obj
    def train(self, x, t, W, B, mlp):

        for epoch in range(self.n_epoch):
            # feedFoward
            a = x
            for layer_nb in range(mlp.nb_layer_):
                i = np.dot(a, mlp.W_[layer_nb]) + mlp.B_[layer_nb]
                a = mlp.tanh_prime(i)


                # # Compute Error and delta
                # for layer_nb in range(mlp.nb_layer_, -1, -1):
                #     if layer_nb == mlp.nb_layer_:
                #         s = np.multiply(np.multiply(np.subtract(d, a), a), np.subtract(1, a))
                #     else:
                #         s = np.multiply(a, np.subtract(1, a))
                #
                #     if layer_nb == 1:
                #         dw = np.
                #     else:
                #         dw =
                #
                # # backprop
                # for layer in reversed(layers):
                #     W = W + dw


def main():
    # Set logging config
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)  # DEBUG to debug, INFO to turn off
    logger = logging.getLogger(__name__)

    load_mlp = input('Open an existing MLP? (yes or no): ')

    if load_mlp == 'yes':
        print('Loading MLP!')
    elif load_mlp == 'no':
        print('Starting new MLP')
    else:
        print('only answer "yes or no" with no caps please.')

    trainer = Trainer()

#    trainer.create_batch("static")


if __name__ == '__main__':
    main()
