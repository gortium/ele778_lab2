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

    def __init__(self):
        # Set logging config
        logging.basicConfig(stream=sys.stderr, level=logging.INFO) # DEBUG to debug, INFO to turn off
        self.logger = logging.getLogger(__name__)

        self.data_manager = data_manager.DataManager()

        # Loading trainer config file
        with open(os.path.join(self.data_manager.paths["abs_config_path"], "trainer.yaml"), "r") as stream:
            self.config = list(yaml.load_all(stream))

        # Hyperparameters from config file
        for index, param in enumerate(self.config):
            if "timeout" in param:
                self.timeout = self.config[index]["timeout"]
                self.logger.info("Timeout set to: %f", self.timeout)
            else:
                self.logger.error("'timeout' parameter not found")
            if "learning_rate" in param:
                self.learning_rate = self.config[index]["learning_rate"]
                self.logger.info("learning_rate set to: %f", self.learning_rate)
            else:
                self.logger.error("'learning_rate' parameter not found")
            if "momentum" in param:
                self.momentum = self.config[index]["momentum"]
                self.logger.info("momentum set to: %f", self.momentum)
            else:
                self.logger.error("'momentum' parameter not found")
            if "train_batch_size" in param:
                self.train_batch_size = self.config[index]["train_batch_size"]
                self.logger.info("train_batch_size set to: %f", self.train_batch_size)
            else:
                self.logger.error("'train_batch_size' parameter not found")
            if "vc_batch_size" in param:
                self.vc_batch_size = self.config[index]["vc_batch_size"]
                self.logger.info("vc_batch_size set to: %f", self.vc_batch_size)
            else:
                self.logger.error("'vc_batch_size' parameter not found")
            if "test_batch_size" in param:
                self.test_batch_size = self.config[index]["test_batch_size"]
                self.logger.info("test_batch_size set to: %f", self.test_batch_size)
            else:
                self.logger.error("'test_batch_size' parameter not found")
            if "nb_epoch" in param:
                self.nb_epoch = self.config[index]["nb_epoch"]
                self.logger.info("nb_epoch set to: %f", self.nb_epoch)
            else:
                self.logger.error("'nb_epoch' parameter not found")
            if "vc_min" in param:
                self.vc_min = self.config[index]["vc_min"]
                self.logger.info("vc_min set to: %f", self.vc_min)
            else:
                self.logger.error("'vc_min' parameter not found")

        # Loading MLP config file
        with open(os.path.join(self.data_manager.paths["abs_config_path"], "mlp.yaml"), "r") as stream:
            self.config = list(yaml.load_all(stream))

        # MLP Hyperparameters from config file
        for index, param in enumerate(self.config):
            if "nb_layer" in param:
                self.nb_layer = self.config[index]["nb_layer"]
                self.logger.info("nb_layer set to: %f", self.nb_layer)
            else:
                self.logger.error("'nb_layer' parameter not found")

            if "nb_input" in param:
                self.nb_input = self.config[index]["nb_input"]
                self.logger.info("nb_input set to: %f", self.nb_input)
            else:
                self.logger.error("'nb_input' parameter not found")

            if "nb_hidden" in param:
                self.nb_hidden = self.config[index]["nb_hidden"]
                self.logger.info("nb_hidden set to: %f", self.nb_hidden)
            else:
                self.logger.error("'nb_hidden' parameter not found")

            if "nb_output" in param:
                self.nb_output = self.config[index]["nb_output"]
                self.logger.info("nb_output set to: %f", self.nb_output)
            else:
                self.logger.error("'nb_output' parameter not found")

            if "activation" in param:
                self.activation = self.config[index]["activation"]
                self.logger.info("activation set to: %s", str(self.activation))
            else:
                self.logger.error("'activation' parameter not found")

            if "filter" in param:
                self.filter = self.config[index]["filter"]
                self.logger.info("filter set to: %s", str(self.filter))
            else:
                self.logger.error("'filter' parameter not found")

        # Variable
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

    # Ask configuration parameter on console and save it in config file
    def input_config(self):

        # Loading MLP config file
        with open(os.path.join(self.data_manager.paths["abs_config_path"], "mlp.yaml"), "r") as stream:
            self.config = list(yaml.load_all(stream))

        self.nb_layer = input('How many layer? (1, 2 or 3): ')
        self.nb_input = input('How many input? (default 720): ')
        self.nb_hidden = input('How many hidden neurones? (default 40): ')
        self.nb_output = input('How many output neurones? (recommend 10): ')
        self.activation = input('Wich activation? (sigmoid or tanh): ')
        self.filter = input('Wich filter type? (static, dynamic or combined): ')

        # MLP Hyperparameters in config file
        for index, param in enumerate(self.config):
            if "nb_layer" in param:
                self.config[index]["nb_layer"] = self.nb_layer
                self.logger.info("nb_layer set to: %f", self.nb_layer)
            else:
                self.logger.error("'nb_layer' parameter not found")

            if "nb_input" in param:
                self.config[index]["nb_input"] = self.nb_input
                self.logger.info("nb_input set to: %f", self.nb_input)
            else:
                self.logger.error("'nb_input' parameter not found")

            if "nb_hidden" in param:
                self.config[index]["nb_hidden"] = self.nb_hidden
                self.logger.info("nb_hidden set to: %f", self.nb_hidden)
            else:
                self.logger.error("'nb_hidden' parameter not found")

            if "nb_output" in param:
                self.config[index]["nb_output"] = self.nb_output
                self.logger.info("nb_output set to: %f", self.nb_output)
            else:
                self.logger.error("'nb_output' parameter not found")

            if "activation" in param:
                self.config[index]["activation"] = self.activation
                self.logger.info("activation set to: %s", str(self.activation))
            else:
                self.logger.error("'activation' parameter not found")

            if "filter" in param:
                self.config[index]["filter"] = self.filter
                self.logger.info("filter set to: %s", str(self.filter))
            else:
                self.logger.error("'filter' parameter not found")

        # TO save in MLP config file
        with open(os.path.join(self.data_manager.paths["abs_config_path"], "mlp.yaml"), "w") as stream:
            yaml.dump(self.config, stream, default_flow_style=False)


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
            if mlp.activation == "sigmoid":
                self.data_manager.find_and(self.data_manager.normalize_list, 0, 1)
            elif mlp.activation == "tanh":
                self.data_manager.find_and(self.data_manager.normalize_list, -1, 1)

        if mode == "train":
            size = self.train_batch_size
            nb_epoch = self.nb_epoch
        elif mode == "vc":
            size = self.vc_batch_size
            nb_epoch = 1
        elif mode == "test":
            size = self.test_batch_size
            nb_epoch = 1

        self.batchs = []
        self.Ys = []
        for i in range(nb_epoch):
            batch = []
            Y = []
            for j in range(size):
                self.data_manager.find_random_and(self.data_tree[mode], self.data_manager.save_obj)
                batch.append([inner for outer in self.data_manager.saved_obj for inner in outer][:mlp.nb_input])

                if mlp.activation == "sigmoid":
                    l = 0
                    h = 1
                elif mlp.activation == "tanh":
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
        starttime = time.time()
        timeout = time.time() + self.timeout
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
            for i in range(self.vc_batch_size):
                good += np.array_equal(self.Ys[0][i], result[i])

            pourcent = good * 100 / self.vc_batch_size

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
        for i in range(self.test_batch_size):
            good += np.array_equal(self.Ys[0][i], result[i])

        pourcent = good * 100 / self.test_batch_size

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

    def load_mlp(self, mlp):
        with open(os.path.join(self.data_manager.paths["abs_project_path"], "save/mlp.pickle"),"rb") as input_file:
            mlp = cPickle.load(input_file)

def main():
    # Set logging config
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)  # DEBUG to debug, INFO to turn off
    logger = logging.getLogger(__name__)

    trainer = Trainer()
    mlp = mlperceptron.MLP(trainer.nb_layer, trainer.nb_input, trainer.nb_hidden, trainer.nb_output, trainer.activation)
    time.sleep(0.05)

    load_mlp = input('Load last MLP? (yes or no): ')

    if load_mlp == 'yes':
        logger.info('Loading MLP!')
        trainer.load_mlp(mlp)
    elif load_mlp == 'no':
        logger.info('Starting new MLP')
        time.sleep(0.05)
        trainer.input_config()
        trainer.train(mlp, trainer.filter)
    else:
        logger.info('only answer "yes" or "no" with no caps please.')

#    mlperceptron.MLP()

if __name__ == '__main__':
    main()
