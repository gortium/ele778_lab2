import logging
import os
import sys
import yaml
import functools
import numpy as np
import data_manager
import mlperceptron
import pickle
import random
from scipy import optimize

class Trainer:

    def __init__(self, *existing_mlp):  # Constructeur
        # Hyperparameters
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.batch_size = 1
        self.n_epoch = 100
        self.activation_function = "sigmoid"

        # Supervising matrix
        D = np.zeros((10, 10), float)
        np.fill_diagonal(D, 1)

        self.data_manager = data_manager.DataManager()
        self.data_tree = None
        self.batch = []
        #self.batch = np.array(self.batch)
        self.goodAnswers = []

        # if existing_mlp:
        #     self.mlp = existing_mlp
        # else:
        #     self.mlp = mlp.MLP()

    # Create all the epoch baths, need to know: combined, static or dynamic data ?
    def create_batch(self, witch_filter, mlp):
        # Load data
        if self.data_tree is None:
            if witch_filter == "combined":
                if self.data_manager.trees["combined"] is None:
                    self.data_manager.fetch_data("combined",
                                                 os.path.join(self.data_manager.paths["abs_filtered_data_path"], "combined"))
                self.data_tree = self.data_manager.trees["combined"]
            elif witch_filter == "static":
                if self.data_manager.trees["static"] is None:
                    self.data_manager.fetch_data("static",
                                                 os.path.join(self.data_manager.paths["abs_filtered_data_path"], "static"))
                self.data_tree = self.data_manager.trees["static"]
            elif witch_filter == "dynamic":
                if self.data_manager.trees["dynamic"] is None:
                    self.data_manager.fetch_data("dynamic",
                                                 os.path.join(self.data_manager.paths["abs_filtered_data_path"], "dynamic"))
                self.data_tree = self.data_manager.trees["dynamic"]
            else:
                return False

            # Normolize the data
            self.data_manager.find_and(self.data_tree, self.data_manager.save_minmax)
            if self.activation_function == "sigmoid":
                self.data_manager.find_and(self.data_manager.normalize_list, 0, 1)
            elif self.activation_function == "tanh":
                self.data_manager.find_and(self.data_manager.normalize_list, -1, 1)

        for batch in range(self.batch_size):
            self.data_manager.find_random_and(self.data_tree, self.data_manager.save_obj)
            self.batch.append([inner for outer in self.data_manager.saved_obj for inner in outer][:mlp.nb_input])

            if self.data_manager.list_name[0] == "0":
                self.goodAnswers.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.data_manager.list_name[0] == "1":
                self.goodAnswers.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            elif self.data_manager.list_name[0] == "2":
                self.goodAnswers.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
            elif self.data_manager.list_name[0] == "3":
                self.goodAnswers.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
            elif self.data_manager.list_name[0] == "4":
                self.goodAnswers.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
            elif self.data_manager.list_name[0] == "5":
                self.goodAnswers.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
            elif self.data_manager.list_name[0] == "6":
                self.goodAnswers.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
            elif self.data_manager.list_name[0] == "7":
                self.goodAnswers.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
            elif self.data_manager.list_name[0] == "8":
                self.goodAnswers.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
            elif self.data_manager.list_name[0] == "9":
                self.goodAnswers.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

        self.batch = np.array(self.batch)
        self.goodAnswers = np.array(self.goodAnswers)

            #self.goodAnswers.append(self.data_manager.list_name.copy())

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad

    def train(self, X, y):
        def __init__(self, N):
            # Make Local reference to network:
            self.N = N

        def callbackF(self, params):
            self.N.setParams(params)
            self.J.append(self.N.costFunction(self.X, self.y))
            self.testJ.append(self.N.costFunction(self.testX, self.testY))

        def costFunctionWrapper(self, params, X, y):
            self.N.setParams(params)
            cost = self.N.costFunction(X, y)
            grad = self.N.computeGradients(X, y)

            return cost, grad

        def train(self, trainX, trainY, testX, testY):
            # Make an internal variable for the callback function:
            self.X = trainX
            self.y = trainY

            self.testX = testX
            self.testY = testY

            # Make empty list to store training costs:
            self.J = []
            self.testJ = []

            params0 = self.N.getParams()

            options = {'maxiter': 200, 'disp': True}
            _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',
                                     args=(trainX, trainY), options=options, callback=self.callbackF)

            self.N.setParams(_res.x)
            self.optimizationResults = _res

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

    trainer = Trainer()

    mlp = mlperceptron.MLP(2, 40, 60*12, 10)

    trainer.create_batch("static", mlp)

    trainer.train()


if __name__ == '__main__':
    main()
