import logging
import os
import sys
import yaml
import functools
import numpy as np
import data_manager
import lvqantization
import _pickle as cPickle
import time
import random
#import matplotlib.pyplot as plt

class Trainer:

    def __init__(self):
        # Set logging config
        logging.basicConfig(stream=sys.stderr, level=logging.INFO) # DEBUG to debug, INFO to turn off
        self.logger = logging.getLogger(__name__)

        self.data_manager = data_manager.DataManager()

        # Loading trainer config file
        with open(os.path.join(self.data_manager.paths["abs_config_path"], "trainer_lvq.yaml"), "r") as stream:
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

        # Loading LVQ config file
        with open(os.path.join(self.data_manager.paths["abs_config_path"], "lvq.yaml"), "r") as stream:
            self.config = yaml.load(stream)

        # LVQ Hyperparameters from config file
        for index, param in enumerate(self.config):
            if "nb_represent" in param:
                self.nb_represent = self.config[index]["nb_represent"]
                self.logger.info("nb_represent set to: %f", self.nb_represent)
            else:
                self.logger.error("'nb_layer' parameter not found")

            if "nb_input" in param:
                self.nb_input = self.config[index]["nb_input"]
                self.logger.info("nb_input set to: %f", self.nb_input)
            else:
                self.logger.error("'nb_input' parameter not found")

            if "nb_classe" in param:
                self.nb_classe = self.config[index]["nb_classe"]
                self.logger.info("nb_classe set to: %f", self.nb_classe)
            else:
                self.logger.error("'nb_classe' parameter not found")

            if "nb_output" in param:
                self.nb_output = self.config[index]["nb_output"]
                self.logger.info("nb_output set to: %f", self.nb_output)
            else:
                self.logger.error("'nb_output' parameter not found")

            if "filter" in param:
                self.filter = self.config[index]["filter"]
                self.logger.info("filter set to: %s", str(self.filter))
            else:
                self.logger.error("'filter' parameter not found")

        # Variable
        self.data_tree = None
        self.batchs = []
        self.W = {row: [] for row in range(self.nb_classe)}
        self.vc_pourcents = []
        self.test_pourcents = []

    # Ask configuration parameter on console and save it in config file
    def input_config(self):

        # Loading LVQ config file
        with open(os.path.join(self.data_manager.paths["abs_config_path"], "lvq.yaml"), "r") as stream:
            self.config_input = yaml.load(stream)

        self.nb_represent = input('How many representatives? (default 6): ')
        self.nb_input = input('How many input? (default 60): ')
        self.nb_classe = input('How many classes? (default 10): ')
        self.nb_output = input('How many output neurones? (recommend 10): ')
        self.filter = input('Wich filter type? (static, dynamic or combined): ')

        # LVQ Hyperparameters in config file
        for index, param in enumerate(self.config):
            if "nb_represent" in param:
                self.config_input[index]["nb_represent"] = int(self.nb_represent)
                self.logger.info("nb_represent set to: %f", float(self.nb_represent))
            else:
                self.logger.error("'nb_layer' parameter not found")

            if "nb_input" in param:
                self.config_input[index]["nb_input"] = int(self.nb_input)
                self.logger.info("nb_input set to: %f", float(self.nb_input))
            else:
                self.logger.error("'nb_input' parameter not found")

            if "nb_classe" in param:
                self.config_input[index]["nb_classe"] = int(self.nb_classe)
                self.logger.info("nb_classe set to: %f", float(self.nb_classe))
            else:
                self.logger.error("'nb_classe' parameter not found")

            if "nb_output" in param:
                self.config_input[index]["nb_output"] = int(self.nb_output)
                self.logger.info("nb_output set to: %f", float(self.nb_output))
            else:
                self.logger.error("'nb_output' parameter not found")

            if "filter" in param:
                self.config_input[index]["filter"] = str(self.filter)
                self.logger.info("filter set to: %s", str(self.filter))
            else:
                self.logger.error("'filter' parameter not found")

        # TO save in LVQ config file
        with open(os.path.join(self.data_manager.paths["abs_config_path"], "lvq.yaml"), "w") as stream:
            yaml.dump(self.config_input, stream, default_flow_style=False)


    # Create all the epoch baths, need to know: combined, static or dynamic data ?
    def create_batch(self, witch_filter, mode, lvq):

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
            self.data_manager.find_and(self.data_manager.normalize_list, 0, 1)

        #   self.data_manager.find_and(self.data_tree, self.data_manager.save_minmax)
        #   if lvq.activation == "sigmoid":
        #       self.data_manager.find_and(self.data_manager.normalize_list, 0, 1)
        #   elif lvq.activation == "tanh":
        #       self.data_manager.find_and(self.data_manager.normalize_list, -1, 1)

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

        for i in range(nb_epoch):
            batch = {row: [] for  row in range(self.nb_classe)}
            for j in range(size):
                self.data_manager.find_random_and(self.data_tree[mode], self.data_manager.save_obj)

                if self.data_manager.list_name[0] == "o":
                    batch[0].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                elif self.data_manager.list_name[0] == "1":
                    batch[1].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                elif self.data_manager.list_name[0] == "2":
                    batch[2].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                elif self.data_manager.list_name[0] == "3":
                    batch[3].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                elif self.data_manager.list_name[0] == "4":
                    batch[4].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                elif self.data_manager.list_name[0] == "5":
                    batch[5].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                elif self.data_manager.list_name[0] == "6":
                    batch[6].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                elif self.data_manager.list_name[0] == "7":
                    batch[7].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                elif self.data_manager.list_name[0] == "8":
                    batch[8].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                elif self.data_manager.list_name[0] == "9":
                    batch[9].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                else:
                    self.logger.debug("Got a rare fish here.. ")

            self.batchs.append(batch)

    #
    def select_representative(self, witch_filter, lvq):

        # Load data
        if self.data_tree is None:
            if witch_filter == "combined":
                if self.data_manager.trees["combined"] is None:
                    self.data_manager.fetch_data("combined",
                                                 os.path.join(self.data_manager.paths["abs_filtered_data_path"],
                                                              "combined"))
                self.data_tree = self.data_manager.trees["combined"]
            elif witch_filter == "static":
                if self.data_manager.trees["static"] is None:
                    self.data_manager.fetch_data("static",
                                                 os.path.join(self.data_manager.paths["abs_filtered_data_path"],
                                                              "static"))
                self.data_tree = self.data_manager.trees["static"]
            elif witch_filter == "dynamic":
                if self.data_manager.trees["dynamic"] is None:
                    self.data_manager.fetch_data("dynamic",
                                                 os.path.join(self.data_manager.paths["abs_filtered_data_path"],
                                                              "dynamic"))
                self.data_tree = self.data_manager.trees["dynamic"]
            else:
                return False

            size = self.train_batch_size
            representor = {row: [] for  row in range(self.nb_classe)}

            while 1:
                for j in range(size):
                    self.data_manager.find_random_and(self.data_tree["train"], self.data_manager.save_obj)

                    if self.data_manager.list_name[0] == "o" and len(representor[0]) < self.nb_represent:
                        representor[0].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                    elif self.data_manager.list_name[0] == "1" and len(representor[1]) < self.nb_represent:
                        representor[1].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                    elif self.data_manager.list_name[0] == "2" and len(representor[2]) < self.nb_represent:
                        representor[2].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                    elif self.data_manager.list_name[0] == "3" and len(representor[3]) < self.nb_represent:
                        representor[3].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                    elif self.data_manager.list_name[0] == "4" and len(representor[4]) < self.nb_represent:
                        representor[4].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                    elif self.data_manager.list_name[0] == "5" and len(representor[5]) < self.nb_represent:
                        representor[5].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                    elif self.data_manager.list_name[0] == "6" and len(representor[6]) < self.nb_represent:
                        representor[6].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                    elif self.data_manager.list_name[0] == "7" and len(representor[7]) < self.nb_represent:
                        representor[7].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                    elif self.data_manager.list_name[0] == "8" and len(representor[8]) < self.nb_represent:
                        representor[8].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                    elif self.data_manager.list_name[0] == "9" and len(representor[9]) < self.nb_represent:
                        representor[9].append([inner for outer in self.data_manager.saved_obj for inner in outer][:lvq.nb_input])
                    else:
                        self.logger.debug("Got a rare fish here.. ")

                # test if all representant are selected
                if sum(len(representor[i]) for i in representor) == (self.nb_represent * self.nb_classe):
                    self.W = representor
                    break


    # input data, transpose, layers, biases, lvq obj
    def train(self, lvq, data_type):
        # plt.axis([0, 10, 0, 1])
        # plt.ion()
        starttime = time.time()
        timeout = time.time() + self.timeout
        self.logger.info("Training begin..")
        pourcent = 0

        # Select representative
        self.select_representative(data_type, lvq)

        while not pourcent > self.vc_min and not time.time() > timeout:
            for epoch in range(self.nb_epoch):

                # Generate batch
                self.create_batch(data_type, "train", lvq)

                # distance_ calculation
                distance = lvq.distance(self.batchs[epoch], self.W)

                # find the closest representative of each input vector of 1 batch
                closest_tr = lvq.closest_in(distance, self.train_batch_size)

                learning_rate = self.learning_rate * (1 - (epoch / self.nb_epoch))

                # Compute push or pull of each closest representative of the input vectors
                self.W = lvq.push_pull(self.batchs[epoch], self.W, closest_tr, learning_rate)

            # **** vc ****
            self.logger.info("All epoch done, now VC")

            # Generate batch
            self.create_batch(data_type, "vc", lvq)

            # Predicting
            closest_vc = lvq.predict(self.batchs[0], self.W, self.vc_batch_size)

            # checking answers
            good = 0
            for i in range(self.vc_batch_size):
                good += np.array_equal(closest_vc[i][0], closest_vc[i][2])

            pourcent = good * 100 / self.vc_batch_size

            self.vc_pourcents.append([pourcent, time.time()])

            self.logger.info("VC result: %f %%", pourcent)

        training_time = (time.time() - starttime)
        self.logger.info("Training took %f seconds", training_time)

        # **** Generalization test ****
        self.logger.info("VC passed, now TEST")

        self.test(lvq, data_type)

        self.logger.info("Saving this beauty..")
        self.save_lvq(lvq)

    def test(self, lvq, data_type):
        # Generate batch
        self.create_batch(data_type, "test", lvq)

        # Predicting
        closest_test = lvq.predict(self.batchs[0], self.W, self.test_batch_size)

        # checking answers
        good = 0
        for i in range(self.test_batch_size):
            good += np.array_equal(closest_test[i][0], closest_test[i][2])

        pourcent = good * 100 / self.test_batch_size

        self.test_pourcents.append([pourcent, time.time()])

        self.logger.info("TEST result: %f %%", pourcent)

    def save_lvq(self, lvq):
        with open(os.path.join(self.data_manager.paths["abs_project_path"], "save/lvq.pickle"), "wb") as output_file:
            cPickle.dump(lvq, output_file)

    def load_lvq(self, lvq, data_type):
        with open(os.path.join(self.data_manager.paths["abs_project_path"], "save/lvq.pickle"),"rb") as input_file:
            lvq = cPickle.load(input_file)

        self.test(lvq, data_type)

def main():
    # Set logging config
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)  # DEBUG to debug, INFO to turn off
    logger = logging.getLogger(__name__)

    trainer = Trainer() # Initialize Class Trainer with LVQ parameters
    lvq = lvqantization.LVQ(trainer.nb_represent, trainer.nb_classe, trainer.nb_input, trainer.nb_output)
    time.sleep(0.05)

    load_lvq = input('Load last lvq? (yes or no): ')
    trainer.train(lvq, trainer.filter)

    if load_lvq == 'yes':
        logger.info('Loading LVQ!')
        time.sleep(0.05)
        trainer.load_lvq(lvq, trainer.filter)
    elif load_lvq == 'no':
        logger.info('Starting new LVQ')
        time.sleep(0.05)
        trainer.input_config()
        trainer.train(lvq, trainer.filter)
    else:
        logger.info('only answer "yes" or "no" with no caps please.')

if __name__ == '__main__':
    main()
