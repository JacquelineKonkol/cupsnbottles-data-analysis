# TODO generalize so different datasets can be used
# TODO falls vortrainierter Classifier: welche Testdaten sollen genommen werden?
# TODO Option, neu trainierten Classifier zu speichern?
import os
import tools.basics as tools
import plotting
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, model_selection
from joblib import dump, load
from sklearn import metrics
import pandas as pd
import tools.settings as settings
################################################################################
####################################specify#####################################

config = settings.config()
classifiers = settings.get_classifiers()
classifier = "nearest_neighbors" # look up in classifier_names list
use_pretrained_classifier = False
imgs_falsely_classified = False # only misclassified images are used in                         #
dims = 2

################################################################################

def prepare_clf(X_train, y_train):
    if use_pretrained_classifier:
        # load the desired trained classifier
        clf = load(config.path_trained_classifiers + config.path_dataset + classifier.replace(' ', '_') + ".joblib")
    else:
        # train anew with best model parameters
        loaded_params = load(config.path_best_params + config.path_dataset + classifier.replace(' ', '_') + "_params.joblib")
        clf = classifiers[config.classifier_names.index(classifier)]
        if classifier != "glvq":
            clf.set_params(**loaded_params)
        clf.fit(X_train, y_train)
    return clf


def visualization(X_test, X_train, y_train, y_test, y_pred_train, y_pred, df, y, label_names, pred_proba, score, filenames):
    # plotting t-SNE
    title = classifier + ', trained on ' + str(len(X_train)) + ' samples. Score: ' + str(score)

    inds = np.array(df.index)
    indices = None
    # plot only misclassifications
    if imgs_falsely_classified:
        indices = np.argwhere(y_pred != y_test).flatten()

        title_imgs = 'Images that were falsely classified by ' + classifier
    # plot random images into the scatter
    else:
        imgs_to_plot = 20
        indices = np.random.randint(0, len(y_test), (imgs_to_plot))
        title_imgs = str(imgs_to_plot) + ' random images'

    if (len(np.unique(y_train)) == len(label_names)):
        cm_train = metrics.confusion_matrix(y_train, y_pred_train)
        plotting.plot_confusion_matrix(cm_train, classes=label_names, img_name="absolute_cupsnbottles_train", cmap=plt.cm.Blues)
        plotting.plot_confusion_matrix(cm_train, classes=label_names, img_name="norm_cupsnbottles_train", normalize=True,
                                       title='Normalized confusion matrix, trainings data', cmap=plt.cm.Blues)

    if (len(np.unique(y_test)) == len(label_names)):
        cm = metrics.confusion_matrix(y_test, y_pred)
        plotting.plot_confusion_matrix(cm, classes=label_names, img_name="absolute_cupsnbottles", cmap=plt.cm.Greens)
        plotting.plot_confusion_matrix(cm, classes=label_names, img_name="norm_cupsnbottles", normalize=True,
                                       title='Normalized confusion matrix', cmap=plt.cm.Greens)


    if (pred_proba is not None):
        print(config.num_samples)
        X_embedded = plotting.t_sne_plot(X_test, y_test, y_pred, pred_proba, label_names, title, config.num_samples,
                                         classifier,
                                         "cupsnbottles", dims)
        imgs = tools.load_images(config.path_dataset, inds[indices], filenames)
        plotting.image_conf_scatter(X_embedded, imgs, indices, title_imgs, pred_proba, classifier)


def create_filter_maske(y_test, y_pred, pred_proba, mode="all_test_samples", confidence_threshold = 0.7):

    if mode == "wrong_test_samples":
        filter = y_test != y_pred
    elif mode == "unconfident_test_samples":
        filter = pred_proba < confidence_threshold
    else:
        filter = np.ones(y_test.shape, dtype=bool)  # default take all test samples
    return filter


def calculate_cluster_mean(X_2dim, y, label_names):
    cluster_infos = {}
    for i in range(len(label_names)):
        X_class_indx =  y == i
        X_selected = X_2dim[X_class_indx]
        mean = np.mean(X_selected, axis=0)
        var = np.var(X_selected, axis=0)
        cluster_infos[i] = {'mean':mean, "var":var}

    return cluster_infos


def analysis(X, y_train, X_test, y_test, y_pred, label_names, pred_proba, pred_proba_all, clf, indx_test):
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})

    X_embedded = tools.t_sne(X)
    cluster_means = calculate_cluster_mean(X_embedded, y_train, label_names)

    filter_mask = create_filter_maske(y_test, y_pred, pred_proba, mode="all_test_samples") # possible modes: all_test_samples, wrong_test_samples, unconfident_test_samples
    true_labelnames = [label_names[i] for i in y_test[filter_mask]]
    predict_labelnames = [label_names[i] for i in y_pred[filter_mask].astype(int)]
    IDs = [indx_test[i] for i, value in enumerate(filter_mask) if value]

    dataFalsePredict = {'IDs': IDs,
                        'True Label':y_test[filter_mask],
                        'True Labelname': true_labelnames,
                        'Predict Label':y_pred[filter_mask].astype(int),
                        'Predict Labelname': predict_labelnames
                        }

    if pred_proba is not None:
        dataFalsePredict['Predict Prob.'] = pred_proba[filter_mask]

    for i in range(0, len(label_names)):
        key ='Dist to Cluster ' + label_names[i]
        dataFalsePredict[key] = ["{:.3f}".format(float(np.linalg.norm(point-cluster_means[i]['mean']))).replace(".", ",") for point in X_embedded[IDs]]
        key = 'Predict Prob. ' + label_names[clf.classes_[i]]
        dataFalsePredict[key] = pred_proba_all[filter_mask][:, clf.classes_[i]]

    df = pd.DataFrame(dataFalsePredict, columns=dataFalsePredict.keys())
    if not os.path.isdir("evaluation"):
        os.mkdir("evaluation")
    df.to_csv("evaluation/" + config.path_dataset.replace('/', '') + "_analysis_" + classifier.replace(' ', '_') + ".csv", mode='w',
                     sep=";", index=False)


def main():
    X, y_encoded, y, label_names, df, filenames = tools.load_gt_data(config.num_samples, config.path_dataset)
    X_train, X_test, y_train, y_test, filenames_train, filenames_test= model_selection.train_test_split(X, y_encoded, filenames, test_size=0.33, random_state=42)

    ### TEMP
    # indicesVanilla, indicesOverlap, indicesAmbiguous, indicesBoth = tools.categorize_data(df)
    # X_train_without_ambiguous = X_train[], y_train[]
    # X_train_only_ambiguous, y_train_only_ambiguous = X_train[indicesAmbiguous], y_train[indicesAmbiguous]
    # X_test_without_ambiguous = X_test[], y_test[]
    # X_test_only_ambiguous = X_test[], y_test[]

    X_train_without_ambiguous = []
    X_train_only_ambiguous = []
    X_test_without_ambiguous = []
    X_test_only_ambiguous = []

    y_train_without_ambiguous = []
    y_train_only_ambiguous = []
    y_test_without_ambiguous = []
    y_test_only_ambiguous = []

    #training = 'mixed'
    #training = 'ambiguous only'
    training = 'without ambiguous'

    #testing = 'mixed'
    testing = 'ambiguous only'
    #testing = 'without ambiguous'

    filenames_train = filenames_train.tolist()
    filenames_test = filenames_test.tolist()

    filenames_train_ambiguous = []
    filenames_test_ambiguous = []

    for i, row in df.iterrows():
        if row['ambiguous'] == 1:
            print('count')
            if row['index'] in filenames_train:
                X_train_only_ambiguous.append(X_train[filenames_train.index(row['index'])])
                y_train_only_ambiguous.append(y_train[filenames_train.index(row['index'])])
                filenames_train_ambiguous.append(filenames[row['index']])
            else:
                X_test_only_ambiguous.append(X_test[filenames_test.index(row['index'])])
                y_test_only_ambiguous.append(y_test[filenames_test.index(row['index'])])
                filenames_test_ambiguous.append(filenames[row['index']])
        elif row['ambiguous'] == 0:
            if row['index'] in filenames_train:
                X_train_without_ambiguous.append(X_train[filenames_train.index(row['index'])])
                y_train_without_ambiguous.append(y_train[filenames_train.index(row['index'])])
            else:
                X_test_without_ambiguous.append(X_test[filenames_test.index(row['index'])])
                y_test_without_ambiguous.append(y_test[filenames_test.index(row['index'])])

    print(y_train_only_ambiguous)
    if training == 'ambiguous only':
        X_train = np.array(X_train_only_ambiguous)
        y_train = np.array(y_train_only_ambiguous)
    elif training == 'without ambiguous':
        X_train = np.array(X_train_without_ambiguous)
        y_train = np.array(y_train_without_ambiguous)
    if testing == 'ambiguous only':
        print(y_train_only_ambiguous)
        X_test = np.array(X_test_only_ambiguous + X_train_only_ambiguous)
        y_test = np.array(y_test_only_ambiguous + y_train_only_ambiguous)
        filenames1 = filenames_test_ambiguous + filenames_train_ambiguous
    elif testing == 'without ambiguous':
        X_test = np.array(X_test_without_ambiguous)
        y_test = np.array(y_test_without_ambiguous)
    ### TEMP

    clf = prepare_clf(X_train, y_train)

    y_pred = clf.predict(X_test).astype('int32')
    y_pred_train = clf.predict(X_train).astype('int32')

    score = clf.score(X_test, y_test)

    if classifier == "glvq":
        pred_proba = clf.predict_proba(X_test)
        pred_proba_all = clf.predict_proba_full_matrix(X_test)
    else:
        pred_proba_all = clf.predict_proba(X_test)
        pred_proba = np.max(pred_proba_all, axis=1)

    analysis(X, y_encoded, X_test, y_test, y_pred, label_names, pred_proba, pred_proba_all, clf, filenames1)
    #visualization(X_test, X_train, y_train, y_test, y_pred_train, y_pred, df, y, label_names, pred_proba, score, filenames)


if __name__ == "__main__":
    main()
