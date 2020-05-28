# TODO allow for each classifier that classification confidence can later be extracted
# TODO generalize so different datasets can be used

import cupsnbottles.load_cupsnbottles as load_cupsnbottles
import tools.basics as tools

print(__doc__)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from joblib import dump, load


classifier = "Nearest Neighbors" # If none is given from classifier_names, then gs is performed on all classifiers.
num_samples = 2179 #at most 2179, default: None
dims = None # number of dimensions to reduce to before training
dims_method = None
#dims_method = 'pca'
#dims_method = 'tsne'

path_dataset = '' # TODO generalize so different datasets can be used
path_trained_classifiers = 'trained_classifiers/' # specify where trained classifiers should be saved to
path_best_params = 'classifiers_best_params/' # specify where best parameters should be saved to


def grid_search(X, y, classifier=None):
    """
    Performs grid search of either a specific or all implemented classifiers and
    saves the trained classifier in /trained_classifiers
    :param: X = dataset
    :param: y = labels
    :param: classifier = some string in classifier_names to specify the model (optional)
    :returns: trained classifier (or list of those) with best parameters
    """

    classifier_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
          "Decision Tree", "Random Forest", "Neural Net", "Naive Bayes", "QDA"]

    # can be adjusted
    parameters = [
        {'n_neighbors': [2, 5, 10], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}, # K Nearest Neighbors
        {'kernel':['linear'], 'C': [1, 5, 10], 'probability': [True]}, # Linear SVM (predict_proba with Platt scaling)
        {'kernel':['rbf'], 'C':[1, 5, 10], 'probability': [True]}, # RBF SVM (predict_proba with Platt scaling)
        {}, # Gaussian Process
        {'max_depth':[None, 5, 10], 'min_samples_split': [2, 5, 10]}, # Decision Tree
        {'max_depth':[None, 5, 10], 'n_estimators':[10, 50, 100], 'max_features':[1]}, # Random Forest
        {'alpha': [0.0001, 0.001], 'max_iter': [1000, 2000]}, # Neural Net
        {}, # Naive Bayes
        #{'var_smoothing': [1e-9]}, # Naive Bayes this does not work eventough it's the default value
        {'reg_param': [0.0, 0.5],'tol': [1.0e-2, 1.0e-4, 1.0e-6]}] # Quadratic Discriminant Analysis

    classifiers = [
         KNeighborsClassifier(),
         SVC(),
         SVC(),
         GaussianProcessClassifier(),
         DecisionTreeClassifier(),
         RandomForestClassifier(),
         MLPClassifier(),
         GaussianNB(),
         QuadraticDiscriminantAnalysis()]


    gs_classifiers = []

    # perform grid search over specified classifier
    if classifier is not None:
        clf_index = classifier_names.index(classifier)
        clf = GridSearchCV(classifiers[clf_index], parameters[clf_index], return_train_score=True)
        print(clf.fit(X, y))
        gs_classifiers.append(clf)
        dump(clf, path_trained_classifiers + classifier.replace(' ', '_') + '.joblib')
        dump(clf.best_params_, path_best_params + classifier.replace(' ', '_') + '_params.joblib')
        result_df = pd.DataFrame.from_dict(clf.cv_results_)
        result_df.insert(len(result_df.columns)-1, "Params", clf.cv_results_['params'], True)
        print(reuslt_df)
        print('The best parameters for ' +  classifier_names[clf_index] + ' are: ', clf.best_params_)
        print('>> DONE')

    # perform grid search over all classifier
    else:
        for i, classifier in enumerate(classifiers):
            clf = GridSearchCV(classifier, parameters[i], return_train_score=True)
            print(clf.fit(X, y))
            gs_classifiers.append(clf)
            dump(clf, path_trained_classifiers + classifier_names[i].replace(' ', '_') + '.joblib')
            dump(clf.best_params_, path_best_params + classifier_names[i].replace(' ', '_') + '_params.joblib')
            result_df = pd.DataFrame.from_dict(clf.cv_results_)
            result_df.insert(len(result_df.columns) - 1, "Params", clf.cv_results_['params'], True)
            print(result_df)
            print('The best parameters for ' +  classifier_names[i] + ' are: ', clf.best_params_)
        print('>> DONE')

    return gs_classifiers


def dim_red(X, dims=2, init='pca'):
    """
    :param: X = dataset
    :param: dims = number of dimensions
    :param: init = either 'pca' or 'tsne'
    """
    if init == 'pca':
        pca = PCA(dims)
        X_embedded = pca.fit_transform(X)

    elif init == 'tsne':
        X_embedded = tools.t_sne(X, dims)
    return X_embedded


def main():
    #X, y, label_names, df = tools.load_gt_data(num_samples)
    # load the data
    X, y_encoded, y, label_names, df = tools.load_gt_data(num_samples)

    if dims is not None:
        if dims_method:
            X = dim_red(X, dims, dims_method)
        else:
            X = dim_red(X, dims)

    gs_classifiers = grid_search(X, y, classifier)


if __name__ == "__main__":
    main()
