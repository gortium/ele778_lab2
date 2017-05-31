import os
import yaml

class PreTraitement: # Définition de notre classe PreTraitrement

    def __init__(self): # Constructeur
        script_dir = os.path.dirname(__file__)
        rel_path = "config/pre_traitement.yaml"
        abs_file_path = os.path.join(script_dir, rel_path)

        stream = open("test", "r")
        docs = yaml.load_all(stream)
        for doc in docs:
            for k, v in doc.items():
                print(k, "->", v)
                print("\n")

        file = open(abs_file_path, 'r')
        line = [map(int, line.split(' ')) for line in file]
        print(line)

        self.test_data_path =

    def fetch_data(self):
        script_dir = os.path.dirname(__file__)
        rel_path = ".." +
        abs_file_path = os.path.join(script_dir, rel_path)

        f = open('input.txt', 'r')
        l = [map(int, line.split(' ')) for line in f]
        print(l)


    def filter_static_energy(self):


    def filter_dynamic_energy(self):


    def filter_combined_energy(self):


def main():
    pre_traitement = PreTraitement()

    pre_traitement.fetch_data()

if __name__ == '__main__':
    main()