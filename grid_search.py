# TODO generalize so different datasets can be used
import itertools

from sklearn import model_selection
import tools.basics as tools
print(__doc__)
import pandas as pd
import tools.settings as settings
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import os
from glvq import *

################################################################################
####################################specify#####################################
config = settings.config()

classifier = None
dims = None # number of dimensions to reduce to before training
dims_method = None
#dims_method = 'pca'
#dims_method = 'tsne'

classifiers = settings.get_classifiers()



################################################################################
def save_grid_search_results(clf, classifier_name):
    result_path_params = os.path.join(config.path_best_params, config.path_dataset)
    if not os.path.isdir(result_path_params):
        os.mkdir(result_path_params)

    result_path_clf= os.path.join(config.path_trained_classifiers, config.path_dataset)
    if not os.path.isdir(result_path_clf):
        os.mkdir(result_path_clf)

    result_df = pd.DataFrame.from_dict(clf.cv_results_)
    result_df.insert(0, "Params", clf.cv_results_['params'], True)
    result_df.to_csv(result_path_params + "grid_search_" + classifier_name.replace(' ', '_') + ".csv", mode='w', sep=";", index=False)
    dump(clf, result_path_clf + classifier_name.replace(' ', '_') + '.joblib')
    dump(clf.best_params_, result_path_params+ classifier_name.replace(' ', '_') + '_params.joblib')

    print('The best parameters for ' +  classifier_name + ' are: ', clf.best_params_, ' with score: ', clf.best_score_)


def run_glvq(X, y):
    # Todo gridsearch
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)

    grid_search_results = []
    clf_index = config.classifier_names.index("glvq")
    for param_set in itertools.product(*config.parameters_grid_search[clf_index].values(), repeat=1):

        clf = glvq(max_prototypes_per_class=int(param_set[0]),
                   learning_rate=int(param_set[1]),
                   strech_factor=int(param_set[2]))


        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        grid_search_results.append((score, param_set, clf))
    grid_search_results_sorted= sorted(grid_search_results, key=lambda tup: tup[1],
           reverse=True)


    result_path_params = os.path.join(config.path_best_params, config.path_dataset)
    if not os.path.isdir(result_path_params):
        os.mkdir(result_path_params)

    result_path_clf = os.path.join(config.path_trained_classifiers, config.path_dataset)
    if not os.path.isdir(result_path_clf):
        os.mkdir(result_path_clf)

    dump(grid_search_results_sorted[0][2], result_path_clf + "glvq" + '.joblib')
    dump(grid_search_results_sorted[0][1], result_path_params + "glvq" + '_params.joblib')

    best_clf = grid_search_results_sorted[0][2]

    print("glvq" + ' best params: ',
          grid_search_results_sorted[0][1])

    print("glvq" + ' with train score: ',
          best_clf.score(X_train, y_train))

    print("glvq" + ' with test score: ',
          best_clf.score(X_test, y_test))


def grid_search(X, y, label_names, classifier=None):
    """
    Performs grid search of either a specific or all implemented classifiers and
    saves the trained classifier in /trained_classifiers
    :param: X = dataset
    :param: y = labels
    :param: classifier = some string in classifier_names to specify the model (optional)
    :returns: trained classifier (or list of those) with best parameters
    """

    gs_classifiers = []

    # perform grid search over specified classifier
    if classifier is not None:
        if (classifier == "glvq"):
            run_glvq(X,y)

        else:
            clf_index = config.classifier_names.index(classifier)
            clf = GridSearchCV(classifiers[clf_index], config.parameters_grid_search[clf_index], return_train_score=True)
            #clf = classifiers[clf_index]
            clf.fit(X, y)
            gs_classifiers.append(clf)
            save_grid_search_results(clf, classifier)
            print('>> DONE')

    # perform grid search over all classifier
    else:
        for i, classifier in enumerate(classifiers):
            if (config.classifier_names[i] == "glvq"):
                run_glvq(X, y)
            else:
                clf = GridSearchCV(classifier, config.parameters_grid_search[i], return_train_score=True)
                clf.fit(X, y)
                gs_classifiers.append(clf)
                save_grid_search_results(clf, config.classifier_names[i])
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
    # load the data
    X, y_encoded, y, label_names, df, filenames = tools.load_gt_data()

    if dims is not None:
        if dims_method:
            X = dim_red(X, dims, dims_method)
        else:
            X = dim_red(X, dims)

    gs_classifiers = grid_search(X, y_encoded, label_names, classifier)

if __name__ == "__main__":
    main()
