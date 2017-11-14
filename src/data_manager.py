import logging
import os
import sys
import yaml
import functools
import random
import numpy as np
import copy

class DataManager:

    def __init__(self):  # Constructeur
        # Set logging config
        logging.basicConfig(stream=sys.stderr, level=logging.INFO) # DEBUG to debug, INFO to turn off
        self.logger = logging.getLogger(__name__)

        # Variables
        self.max_value = 0
        self.min_value = 0
        self.list_name = None
        self.saved_obj = None
        self.paths = {}
        self.trees = dict.fromkeys(["raw", "static", "dynamic", "combined"])

        # Generating project and config absolute path
        self.paths["abs_script_path"] = os.path.dirname(__file__).replace("/",os.sep)
        self.paths["rel_config_path"] = "config"
        self.paths["abs_project_path"] = self.paths["abs_script_path"][:-len(self.paths["abs_script_path"].split(os.sep)[-1])]
        self.paths["abs_config_path"] = os.path.join(self.paths["abs_project_path"], self.paths["rel_config_path"])

        # Loading config file
        with open(os.path.join(self.paths["abs_config_path"], "data_manager.yaml"), "r") as stream:
            self.config = list(yaml.load_all(stream))

        # Generating data absolute paths from config file
        for index, param in enumerate(self.config):
            if "raw_data_path" in param:
                self.paths["abs_raw_data_path"] = os.path.join(self.paths["abs_project_path"], self.config[index]["raw_data_path"]).replace("/",os.sep)
                self.logger.debug("raw_data_path: %s", str(self.paths["abs_raw_data_path"]))
            else:
                self.logger.error("'raw_data_path' parameter not found")

            if "filtered_data_path" in param:
                self.paths["abs_filtered_data_path"] = os.path.join(self.paths["abs_project_path"], self.config[index]["filtered_data_path"]).replace("/",os.sep)
                self.logger.debug("filtered_data_path: %s", str(self.paths["abs_filtered_data_path"]))
            else:
                self.logger.error("'filtered_data_path' parameter not found")

###########################
    ## IO functions ##
###########################

    def fetch_data(self, tree, path):
        # Loading data
        fo = {}
        rootdir = path.rstrip(os.sep)
        start = rootdir.rfind(os.sep) + 1

        # if there is directory, file in this path
        for path, dirs, files in os.walk(rootdir):
            folders = path[start:].split(os.sep)
            subdir = dict.fromkeys(files)

            # If there is files
            for filename in subdir:

                # Open it by line
                with open(path + os.sep + filename) as content_file:
                    content = content_file.read().split("\n")

                    # Create list of line. Each line is a list of number
                    for index, line in enumerate(content):
                        content[index] = content[index].split(" ")
                        # Pop the last 'number' if empty
                        if content[index][-1] == '':
                            content[index].pop()
                    # Same for the last 'line'
                    if not content[-1]:
                        content.pop()
                    # Convert str to float
                    for index1, inner in enumerate(content):
                        for index2, string in enumerate(inner):
                            content[index1][index2] = float(string)
                    subdir[filename] = content

            parent = functools.reduce(dict.get, folders[:-1], fo)
            parent[folders[-1]] = subdir

        # And we have our full directory structure + data !!
        self.trees[tree] = fo[path.split(os.sep)[-len(folders)]]

    # Recreate a folder structure containing data files
    def write_tree(self, root_directory, obj_dict):

        # If you specified a folder in config that do not exist, i handle it here..
        if not os.path.exists(root_directory):
            os.makedirs(root_directory)

        # For every item in the object,
        for key, item in obj_dict.items():

            # check if it a directory or a file
            if not isinstance(item, dict):

                # it's a file, so write the file
                with open(os.path.join(root_directory, key), 'w') as thefile:
                    for line in item:
                        thefile.write((' '.join('{:e}'.format(number) for number in line)) + '\n')

            # it's another dict, so go deeper
            else:
                new_root = os.path.join(root_directory, key)
                if not os.path.exists(new_root):
                    os.makedirs(new_root)
                self.write_tree(new_root, item)

###########################


############################
    ## First functions ##
############################

    # Dive in a nested dictionary until it found a list, than pass it to the callback
    def find_and(self, obj, callback=None, *args):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(obj[k], list):
                    self.list_name = k
                self.find_and(v, callback, *args)
        elif isinstance(obj, list):
            callback(obj, *args)
        else:
            return False

    # Dive randomly in a nested dictionary until it found a list, than return it
    def find_random_and(self, obj, callback=None, *args):
        if isinstance(obj, dict):
                rand_key = random.choice(list(obj.keys()))
                if isinstance(obj[rand_key], list):
                    self.list_name = rand_key
                self.find_random_and(obj[rand_key], callback, *args)
        elif isinstance(obj, list):
            callback(obj, *args)
        else:
            return False

############################


################################
    ## Callback functions ##
################################

    def save_minmax(self, one_list, callback=None, *args):
        for line in one_list:
            for element in line:
                if element < self.min_value:
                    self.min_value = element
                if element > self.max_value:
                    self.max_value = element

        # If there is another Callback, execute it
        if callback is not None:
            callback(one_list, *args)

    def normalize_list(self, one_list, norm_min, norm_max, callback=None, *args):
        for index1, line in enumerate(one_list):
            for index2, element in enumerate(line):
                one_list[index1][index2] = norm_min +\
                                           (element - self.min_value)*(norm_max - norm_min) /\
                                           (self.max_value-self.min_value)

        # If there is another Callback, execute it
        if callback is not None:
            callback(one_list, *args)

    # Order a list of list by the list[][index] from the the highest to the smallest
    def order_by(self, one_list, index, callback=None, *args):
        one_list.sort(key=lambda elem: float(elem[index]), reverse=True)

        # If there is another Callback, execute it
        if callback is not None:
            callback(one_list, *args)

    # Order a list of list by the average of list[][index1] and list[][index2] from the the highest to the smallest
    def order_by_average_of(self, one_list, index1, index2, callback=None, *args):
        one_list.sort(key=lambda elem: ((float(elem[index1]) + float(elem[index2])) / 2), reverse=True)

        # If there is another Callback, execute it
        if callback is not None:
            callback(one_list, *args)

    # Order a list of list by the list[][index] from the the highest to the smallest
    def only_keep(self, one_list, witch_filter, callback=None, *args):
        for line in one_list:
            if witch_filter == "combined":
                del line[25]
                del line[12]
            elif witch_filter == "static":
                del line[12:]
            elif witch_filter == "dynamic":
                del line[25]
                del line[:13]

        # If there is another Callback, execute it
        if callback is not None:
            callback(one_list, *args)

    def np_concatonate(self, one_list, np_array, callback=None, *args):
        np.concatenate((np_array, np.array(one_list)), axis=0)

        # If there is another Callback, execute it
        if callback is not None:
            callback(one_list, *args)

    def save_obj(self, obj, callback=None, *args):
        self.saved_obj = obj.copy()

        # If there is another Callback, execute it
        if callback is not None:
            callback(obj, *args)

    def extrapolate_data(self, one_list, ref_index, callback=None, *args):
        min_dif = []

        # Check if there is enought data
        if len(one_list) < 60:
            self.logger.info("Extrapolating data in file with less than 60 feature")
            missing_data = 60 - len(one_list)

            # For every line..
            for index, line in enumerate(one_list):
                # Skip first line
                if not index == 0:
                    # Creating mapping of diff of the ref_index across line
                    min_dif.append(one_list[index-1][ref_index] - one_list[index][ref_index])

            # For each missing data..
            while missing_data > 0:
                line = []
                # Creating extrapolated line
                for index, number in enumerate(one_list[min_dif.index(min(min_dif))]):
                    line.append(((one_list[min_dif.index(min(min_dif)) - 1][index]) + number) / 2)
                one_list.insert(min_dif.index(min(min_dif)), line)
                min_dif[min_dif.index(min(min_dif))] = max(min_dif)
                missing_data -= 1

        # If there is another Callback, execute it
        if callback is not None:
            callback(one_list, *args)

################################


################################
    ## Filtering functions ##
################################

    # Will create a folder structure containing filtered data by static energy
    def filter_static_energy(self):

        # Fetch original data
        temp_dict = copy.deepcopy(self.trees["raw"])

        # Order it, than extrapolate data
        self.find_and(temp_dict, self.order_by, 12, self.extrapolate_data, 12, self.only_keep, "static")

        # Save filtered data
        self.trees["static"] = temp_dict

    # Will create a folder structure containing filtered data by dynamic energy
    def filter_dynamic_energy(self):

        # Fetch original data
        temp_dict = copy.deepcopy(self.trees["raw"])

        # Order it, than extrapolate data
        self.find_and(temp_dict, self.order_by, 25, self.extrapolate_data, 25, self.only_keep, "dynamic")

        # Save filtered data
        self.trees["dynamic"] = temp_dict

    # Will create a folder structure containing filtered data by a average of static and dynamic energy
    def filter_combined_energy(self):

        # Fetch original data
        temp_dict = copy.deepcopy(self.trees["raw"])

        # Order it, than extrapolate data
        self.find_and(temp_dict, self.order_by_average_of, 12, 25, self.extrapolate_data, 12, self.only_keep, "combined")

        # Save filtered data
        self.trees["combined"] = temp_dict


################################

def main():
    # Set logging config
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)  # DEBUG to debug, INFO to turn off
    logger = logging.getLogger(__name__)

    logger.info("Lets see..")
    data_manager = DataManager()

    logger.info("All right! I have some work to do here..")

    logger.info("Fetching data..")
    data_manager.fetch_data("raw", data_manager.paths["abs_raw_data_path"])

    logger.info("Filtering data by static energy..")
    data_manager.filter_static_energy()
    logger.info("Filtering data by dynamic energy..")
    data_manager.filter_dynamic_energy()
    logger.info("Filtering data by average of static and dynamic energy..")
    data_manager.filter_combined_energy()

    logger.info("Saving filtered static data to file..")
    data_manager.write_tree(os.path.join(data_manager.paths["abs_filtered_data_path"], "static"), data_manager.trees["static"])
    logger.info("Saving filtered dynamic data to file..")
    data_manager.write_tree(os.path.join(data_manager.paths["abs_filtered_data_path"], "dynamic"), data_manager.trees["dynamic"])
    logger.info("Saving filtered combined data to file..")
    data_manager.write_tree(os.path.join(data_manager.paths["abs_filtered_data_path"], "combined"), data_manager.trees["combined"])

    logger.info("DONE, have fun ;)")

if __name__ == '__main__':
    main()
