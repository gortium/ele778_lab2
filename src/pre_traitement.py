import logging
import os
import sys
import yaml
import functools


class PreTraitement:  # Definition de notre classe PreTraitrement

    def __init__(self):  # Constructeur

        # Set logging config
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)  # DEBUG, NOTSET
        self.logger = logging.getLogger(__name__)

        # Variables
        self.paths = {}
        self.raw_tree = {}
        self.filtered_tree = {}

        # Generating project and config absolute path
        self.paths["abs_script_path"] = os.path.dirname(__file__)
        self.paths["rel_config_path"] = "config/pre_traitement.yaml"
        self.paths["abs_project_path"] = self.paths["abs_script_path"][:-len(self.paths["abs_script_path"].split("/")[-1])]
        self.paths["abs_config_path"] = os.path.join(self.paths["abs_project_path"], self.paths["rel_config_path"])

        # Loading config file
        with open(self.paths["abs_config_path"], "r") as stream:
            self.config = list(yaml.load_all(stream))

        # Generating data absolute paths
        if any("data_path" in param for param in self.config):
            for index, param in enumerate(self.config):
                if "data_path" in param:
                    self.paths["abs_data_path"] = os.path.join(self.paths["abs_project_path"], self.config[index]["data_path"])
                    self.logger.debug("data path: %s", str(self.paths["abs_data_path"]))
        else:
            print ("'data path' parameter not found")

    def fetch_data(self):
        # Loading data
        fo = {}
        rootdir = self.paths["abs_data_path"].rstrip(os.sep)
        start = rootdir.rfind(os.sep) + 1
        for path, dirs, files in os.walk(rootdir):
            folders = path[start:].split(os.sep)
            subdir = dict.fromkeys(files)
            for filename in subdir:
                with open(path + "/" + filename) as content_file:
                    content = content_file.read().split("\n")
                    for index, line in enumerate(content):
                        content[index] = content[index].split(" ")
                    subdir[filename] = content
            parent = functools.reduce(dict.get, folders[:-1], fo)
            parent[folders[-1]] = subdir
        self.raw_tree = fo

    # TODO
    # def filter_static_energy(self):
    #
    # TODO
    # def filter_dynamic_energy(self):
    #
    # TODO
    # def filter_combined_energy(self):
    #
    # TODO Fonction pour recreer la structure de folder
    # def walk(root_directory, obj_dict):
    #     for k, v in obj_dict.iteritems():
    #         if is_data(v):
    #             # write the file
    #             write_data(root_directory, k, v)
    #         else:  # it's another dict, so recurse
    #             # add the key to the path
    #             new_root = os.path.join(root_directory, k)  # you'll need to import os
    #             walk(new_root, v)


def main():
    pre_traitement = PreTraitement()

    pre_traitement.fetch_data()

if __name__ == '__main__':
    main()
