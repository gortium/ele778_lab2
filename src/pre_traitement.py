import logging
import os
import sys
import yaml
import functools


class PreTraitement:  # Definition de notre classe PreTraitrement

    def __init__(self):  # Constructeur
        # Set logging config
        logging.basicConfig(stream=sys.stderr, level=logging.INFO) # DEBUG to debug, INFO to turn off
        self.logger = logging.getLogger(__name__)

        # Variables
        self.paths = {}
        self.raw_tree = {}
        self.filtered_static_tree = {}
        self.filtered_dynamic_tree = {}
        self.filtered_combined_tree = {}

        # Generating project and config absolute path
        self.paths["abs_script_path"] = os.path.dirname(__file__)
        self.paths["rel_config_path"] = "config/pre_traitement.yaml"
        self.paths["abs_project_path"] = self.paths["abs_script_path"][:-len(self.paths["abs_script_path"].split("/")[-1])]
        self.paths["abs_config_path"] = os.path.join(self.paths["abs_project_path"], self.paths["rel_config_path"])

        # Loading config file
        with open(self.paths["abs_config_path"], "r") as stream:
            self.config = list(yaml.load_all(stream))

        # Generating data absolute paths from config file
        for index, param in enumerate(self.config):
            if "raw_data_path" in param:
                self.paths["abs_raw_data_path"] = os.path.join(self.paths["abs_project_path"], self.config[index]["raw_data_path"])
                self.logger.debug("raw_data_path: %s", str(self.paths["abs_raw_data_path"]))
            else:
                self.logger.error("'raw_data_path' parameter not found")

            if "filtered_data_path" in param:
                self.paths["abs_filtered_data_path"] = os.path.join(self.paths["abs_project_path"], self.config[index]["filtered_data_path"])
                self.logger.debug("filtered_data_path: %s", str(self.paths["abs_filtered_data_path"]))
            else:
                self.logger.error("'filtered_data_path' parameter not found")

    def fetch_data(self):
        # Loading data
        fo = {}
        rootdir = self.paths["abs_raw_data_path"].rstrip(os.sep)
        start = rootdir.rfind(os.sep) + 1

        # if there is directory, file in this path
        for path, dirs, files in os.walk(rootdir):
            folders = path[start:].split(os.sep)
            subdir = dict.fromkeys(files)

            # If there is files
            for filename in subdir:

                # Open it by line
                with open(path + "/" + filename) as content_file:
                    content = content_file.read().split("\n")

                    # Create list of line. Each line is a list of number
                    for index, line in enumerate(content):
                        content[index] = content[index].split(" ")
                        # Pop the last 'number' as it's a empty one and cause problem in the sort later
                        content[index].pop()
                    # Same for the last 'line'
                    content.pop()
                    subdir[filename] = content
                    for index1, inner in enumerate(content): # convertie les lists en float pour filtrer
                        for index2, string in enumerate(inner):
                            content[index1][index2] = float(string)
            parent = functools.reduce(dict.get, folders[:-1], fo)
            parent[folders[-1]] = subdir

        # And we have our full directory structure + data !!
        self.raw_tree = fo["raw"]

    # Dive in a nested dictionary until it found a list, than pass it to the callback
    def found_and(self, obj, callback=None, *args):
        if isinstance(obj, dict):
            for k, v in obj.items():
                self.found_and(v, callback, *args)
        elif isinstance(obj, list):
            callback(obj, *args)
        else:
            return False

    # Order a list of list by the list[][index] from the the highest to the smallest
    @staticmethod
    def order_by(one_list, index):
        one_list.sort(key=lambda elem: float(elem[index]), reverse=True)

    # Order a list of list by the average of list[][index1] and list[][index2] from the the highest to the smallest
    @staticmethod
    def order_by_average_of(one_list, index1, index2):
        one_list.sort(key=lambda elem: ((float(elem[index1]) + float(elem[index2])) / 2), reverse=True)

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
                        thefile.write(' '.join(str(number) for number in line))

            # it's another dict, so go deeper
            else:
                new_root = os.path.join(root_directory, key)
                if not os.path.exists(new_root):
                    os.makedirs(new_root)
                self.write_tree(new_root, item)

    def extrapolate_data(self, one_file, ref_index):
        min_index = []

        # Check if there is enought data
        if len(one_file) < 60:
            self.logger.info("Extrapolating data in file with less than 60 feature")
            missing_data = 60 - len(one_file)

            # For every line..
            for index, line in enumerate(one_file):
                # Skip first line
                if not index == 0:
                    # Creating mapping of diff of the ref_index across line
                    min_index.append(float(one_file[index-1][ref_index]) - float(one_file[index][ref_index]))

            # For each missing data..
            while missing_data > 0:
                line = []
                # Creating extrapolated line
                for index, number in enumerate(one_file[min_index.index(min(min_index))]):
                    line.append('{:e}'.format(float(one_file[index - 1][index]) - float(number)))
                missing_data -= 1
                min_index.pop(0)
                one_file.insert(min_index.index(min(min_index)), line)

    # Will create a folder structure containing filtered data by static energy
    def filter_static_energy(self):

        # Fetch original data
        temp_dict = self.raw_tree.copy()

        # Order it
        self.found_and(temp_dict, self.order_by, 12)

        # Extrapolate data
        self.found_and(temp_dict, self.extrapolate_data, 12)

        # Save filtered data
        self.filtered_static_tree = temp_dict

        # Write to files
        self.write_tree(os.path.join(self.paths["abs_filtered_data_path"], "static"), self.filtered_static_tree)

    # Will create a folder structure containing filtered data by dynamic energy
    def filter_dynamic_energy(self):

        # Fetch original data
        temp_dict = self.raw_tree.copy()

        # Order it
        self.found_and(temp_dict, self.order_by, 25)

        # Extrapolate data
        self.found_and(temp_dict, self.extrapolate_data, 25)

        # Save filtered data
        self.filtered_dynamic_tree = temp_dict

        # Write to files
        self.write_tree(os.path.join(self.paths["abs_filtered_data_path"], "dynamic"), self.filtered_dynamic_tree)

    # Will create a folder structure containing filtered data by a average of static and dynamic energy
    def filter_combined_energy(self):

        # Fetch original data
        temp_dict = self.raw_tree.copy()

        # Order it
        self.found_and(temp_dict, self.order_by_average_of, 12, 25)

        # Extrapolate data
        self.found_and(temp_dict, self.extrapolate_data, 12)

        # Save filtered data
        self.filtered_combined_tree = temp_dict

        # Write to files
        self.write_tree(os.path.join(self.paths["abs_filtered_data_path"], "combined"), self.filtered_combined_tree)


def main():
    # Set logging config
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)  # DEBUG to debug, INFO to turn off
    logger = logging.getLogger(__name__)

    logger.info("Lets see..")
    pre_traitement = PreTraitement()

    logger.info("All right! I have some work to do here..")

    logger.info("Fetching data..")
    pre_traitement.fetch_data()

    logger.info("Filtering data by static energy..")
    pre_traitement.filter_static_energy()
    logger.info("Filtering data by dynamic energy..")
    pre_traitement.filter_dynamic_energy()
    logger.info("Filtering data by average of static and dynamic energy..")
    pre_traitement.filter_combined_energy()

    logger.info("DONE, have fun ;)")

if __name__ == '__main__':
    main()
