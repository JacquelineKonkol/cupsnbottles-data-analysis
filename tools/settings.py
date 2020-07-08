import configparser
import os
from sklearn import *
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from glvq import *
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
            # [STANDARD]
            self.use_pretrained_classifier = True if config['STANDARD']['use_pretrained_classifier'] == 'True' else False
            self.path_dataset = config['STANDARD']['path_dataset']
            self.num_samples = int(config['STANDARD']['num_samples'])
            self.path_best_params = config['STANDARD']['path_best_params']
            self.path_trained_classifiers = config['STANDARD']['path_trained_classifier']

            # [DATASET]

            self.normal_evaluation = True if config['DATASET']['normal_evaluation'] == 'True' else False
            self.training = config['DATASET']['training']
            self.testing = config['DATASET']['testing']
            self.vanilla_train_part =  float(config['DATASET']['vanilla_train_part'])
            self.vanilla_test_part = float(config['DATASET']['vanilla_test_part'])
            self.ambiguous_train_part =  float(config['DATASET']['ambiguous_train_part'])
            self.ambiguous_test_part = float(config['DATASET']['ambiguous_test_part'])
            self.overlap_train_part =  float(config['DATASET']['overlap_train_part'])
            self.overlap_test_part = float(config['DATASET']['overlap_test_part'])
            self.both_train_part =  float(config['DATASET']['both_train_part'])
            self.both_test_part = float(config['DATASET']['both_test_part'])

            if self.training == "without ambiguous":
                self.ambiguous_train_part = 0.0
            elif self.training == "only ambiguous":
                self.ambiguous_train_part = 1.0
                self.vanilla_train_part = 0.0
                self.overlap_train_part = 0.0
                self.both_train_part = 0.0
            else:
                pass
            if self.testing == "without ambiguous":
                self.ambiguous_test_part = 0.0
            elif self.testing == "only ambiguous":
                self.ambiguous_test_part = 1.0
                self.vanilla_test_part = 0.0
                self.overlap_test_part = 0.0
                self.both_test_part = 0.0
            else:
                pass

            # [GRID SEARCH]
            self.classifier_names = []
            self.parameters_grid_search = []
            for key in config['GRID_SEARCH']:
                # all keys are lowercase
                self.classifier_names.append(key)
                self.parameters_grid_search.append(literal_eval(config['GRID_SEARCH'][key]))

        else:
            raise Exception("Config file: %s does not exists" % config_loc)



def get_classifiers():
    #todo dict statt über position im array
    classifiers = [
         KNeighborsClassifier(),
         SVC(),
         SVC(),
         GaussianProcessClassifier(),
         DecisionTreeClassifier(),
         RandomForestClassifier(),
         MLPClassifier(),
         GaussianNB(),
         QuadraticDiscriminantAnalysis(),
         glvq()
    ]
    return classifiers