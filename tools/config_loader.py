import configparser
import os
from ast import literal_eval


class config():

    def __init__(self, config_loc="./config.ini"):
        """Constructor takes some additional arguments like max_prototypes_per_class, learning_rate. Additionally a
        placement strategy can passed as a callback. The function takes one sample x and corresponding label y and
        returns True if a new prototype should be placed otherwise false. Then the prototype is moved (see fit sample)."""
        if os.path.isfile(config_loc):
            config = configparser.ConfigParser()
            config.read(config_loc)

            # load config
            self.path_dataset = config['STANDARD']['path_dataset']
            self.num_samples = int(config['STANDARD']['num_samples'])
            self.path_best_params = config['STANDARD']['path_best_params']
            self.path_trained_classifiers = config['STANDARD']['path_trained_classifiers']

            self.classifier_names = []
            self.parameters_grid_search = []
            for key in config['GRID_SEARCH']:
                # all keys are lowercase
                self.classifier_names.append(key)
                self.parameters_grid_search.append(literal_eval(config['GRID_SEARCH'][key]))

        else:
            raise Exception("Config file: %s does not exists" % config_loc)



