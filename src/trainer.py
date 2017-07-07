import logging
import os
import sys
import yaml
import functools
import trainer.py

class trainer:  # Definition de notre classe PreTraitrement

    def __init__(self):  # Constructeur
        # Hyperparameters
        learning_rate = 0.01
        momentum = 0.9

    # input data, transpose, layer1, layer2, biases
    def train(self, mlp):

        def feedFoward(self):
            for layer in mlp.W:
                i = np.dot(x, V) + bv
                A += mlp.tanh_prime(i)

        def computeError(self):

        def computeDelta(self):
            for


        def backprop(self):
            for layer in reversed(layers):
                W = W + dw

        for epoch in range(n_epoch):
            self.feedFoward()
            self.computeError()
            self.computeDelta()
            self.backprop()


def main()
    # Set logging config
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)  # DEBUG to debug, INFO to turn off
    logger = logging.getLogger(__name__)





if __name__ == '__main__':
    main()