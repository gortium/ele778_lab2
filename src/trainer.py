import logging
import os
import sys
import yaml
import functools
import numpy as np
import data_manager
import mlperceptron
import _pickle as cPickle
import time
#import matplotlib.pyplot as plt

class Trainer:

    def __init__(self, *existing_mlp):
        # Set logging config
        logging.basicConfig(stream=sys.stderr, level=logging.INFO) # DEBUG to debug, INFO to turn off
        self.logger = logging.getLogger(__name__)

        # Hyperparameters
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.batch_size = 10
        self.vc_size = 100
        self.test_size = 800
        self.nb_epoch = 20
        self.activation = "sigmoid"
        self.vc_min = 85

        # Variable
        self.data_manager = data_manager.DataManager()
        self.data_tree = None
        self.batchs = []
        self.Ys = []
        self.E = []
        self.dW = []
        self.vc_pourcents = []
        self.test_pourcents = []

        # if existing_mlp:
        #     self.mlp = existing_mlp
        # else:
        #     self.mlp = mlp.MLP()

    # Create all the epoch baths, need to know: combined, static or dynamic data ?
    def create_batch(self, witch_filter, mode, mlp):
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
            if self.activation == "sigmoid":
                self.data_manager.find_and(self.data_manager.normalize_list, 0, 1)
            elif self.activation == "tanh":
                self.data_manager.find_and(self.data_manager.normalize_list, -1, 1)

        if mode == "train":
            size = self.batch_size
            nb_epoch = self.nb_epoch
        elif mode == "vc":
            size = self.vc_size
            nb_epoch = 1
        elif mode == "test":
            size = self.test_size
            nb_epoch = 1

        self.batchs = []
        self.Ys = []
        for i in range(nb_epoch):
            batch = []
            Y = []
            for j in range(size):
                self.data_manager.find_random_and(self.data_tree[mode], self.data_manager.save_obj)
                batch.append([inner for outer in self.data_manager.saved_obj for inner in outer][:mlp.nb_input])

                if self.activation == "sigmoid":
                    l = 0
                    h = 1
                elif self.activation == "tanh":
                    l = -1
                    h = 1

                if self.data_manager.list_name[0] == "0" or self.data_manager.list_name[0] == "o":
                    Y.append([h, l, l, l, l, l, l, l, l, l])
                elif self.data_manager.list_name[0] == "1":
                    Y.append([l, h, l, l, l, l, l, l, l, l])
                elif self.data_manager.list_name[0] == "2":
                    Y.append([l, l, h, l, l, l, l, l, l, l])
                elif self.data_manager.list_name[0] == "3":
                    Y.append([l, l, l, h, l, l, l, l, l, l])
                elif self.data_manager.list_name[0] == "4":
                    Y.append([l, l, l, l, h, l, l, l, l, l])
                elif self.data_manager.list_name[0] == "5":
                    Y.append([l, l, l, l, l, h, l, l, l, l])
                elif self.data_manager.list_name[0] == "6":
                    Y.append([l, l, l, l, l, l, h, l, l, l])
                elif self.data_manager.list_name[0] == "7":
                    Y.append([l, l, l, l, l, l, l, h, l, l])
                elif self.data_manager.list_name[0] == "8":
                    Y.append([l, l, l, l, l, l, l, l, h, l])
                elif self.data_manager.list_name[0] == "9":
                    Y.append([l, l, l, l, l, l, l, l, l, h])
                else:
                    self.logger.debug("Got a rare fish here.. ")

            batch = np.array(batch, dtype=float)
            Y = np.array(Y, dtype=float)

            self.batchs.append(batch)
            self.Ys.append(Y)

    # input data, transpose, layers, biases, mlp obj
    def train(self, mlp, data_type):
        # plt.axis([0, 10, 0, 1])
        # plt.ion()
        timeout = time.time() + 60 * 60  # 60 minutes from now
        starttime = time.time()
        self.logger.info("Training begin..")
        pourcent = 0
        while not pourcent > self.vc_min and not time.time() > timeout:
            for epoch in range(self.nb_epoch):

                # Generate batch
                self.create_batch(data_type, "train", mlp)

                # feedFoward
                yhat = mlp.feed_forward(self.batchs[epoch])

                # given activation of the last layer.. the result is..
                result = mlp.max_in(yhat)

                # if not good, LEARN !
                if not np.array_equal(result, self.Ys[epoch]):

                    # Compute delta
                    mlp.backprop(yhat, self.Ys[epoch], self.learning_rate, self.momentum)

            # **** vc ****
            self.logger.info("All epoch done, now VC")

            # Generate batch
            self.create_batch(data_type, "vc", mlp)

            # Predicting
            result = mlp.predict(self.batchs[0])

            # checking answers
            good = 0
            for i in range(self.vc_size):
                good += np.array_equal(self.Ys[0][i], result[i])

            pourcent = good * 100 / self.vc_size

            self.vc_pourcents.append([pourcent, time.time()])

            self.logger.info("VC result: %f %%", pourcent)

        training_time = (time.time() - starttime)
        self.logger.info("Training took %f seconds", training_time)

        # **** Generalization test ****
        self.logger.info("VC passed, now TEST")

        # Generate batch
        self.create_batch(data_type, "test", mlp)

        # Predicting
        result = mlp.predict(self.batchs[0])

        # checking answers
        good = 0
        for i in range(self.test_size):
            good += np.array_equal(self.Ys[0][i], result[i])

        pourcent = good * 100 / self.test_size

        self.test_pourcents.append([pourcent, time.time()])

        self.logger.info("TEST result: %f %%", pourcent)

        self.logger.info("Saving this beauty..")
        self.save_mlp(mlp)

        #     plt.pause(0.05)
        #
        # while True:
        #     plt.pause(0.05)

    def save_mlp(self, mlp):
        with open(os.path.join(self.data_manager.paths["abs_project_path"], "save/mlp.pickle"), "wb") as output_file:
            cPickle.dump(mlp, output_file)


def main():
    # Set logging config
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)  # DEBUG to debug, INFO to turn off
    logger = logging.getLogger(__name__)

    trainer = Trainer()

    mlp = mlperceptron.MLP(3, 60*12, 40,  10, "tanh")

    trainer.train(mlp, "combined")


if __name__ == '__main__':
    main()
