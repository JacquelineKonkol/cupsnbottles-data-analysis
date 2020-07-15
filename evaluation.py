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
imgs_falsely_classified = False # only misclassified images are used in                         #
dims = 2

################################################################################

def prepare_clf(X_train, y_train):
    if config.use_pretrained_classifier:
        # load the desired trained classifier
        clf = load(config.path_trained_classifiers + config.path_dataset + classifier.replace(' ', '_') + ".joblib")
    else:
        # train anew with best model parameters
        #loaded_params = load(config.path_best_params + config.path_dataset + classifier.replace(' ', '_') + "_params.joblib")
        clf = classifiers[config.classifier_names.index(classifier)]
        #if classifier != "glvq":
            #clf.set_params(**loaded_params)
        clf.fit(X_train, y_train)
    return clf


def visualization(X, X_test, X_train, y_train, y_test, y_pred_train, y_pred, df, y, label_names, pred_proba, score, filenames, filenames_train, filenames_test):
    """
    Produces visualization of - confusion matrix for train and test set each (+ normalized version)
                              - scatterplot collage with classification results and useful information
                              - image scatterplots of categories of interest
    :param X_test
    :param X_train
    :param y_train
    :param y_test
    :param y_pred_train:
    :param y_pred:
    :param df: pandas dataframe of the meta-data given in properties.csv
    :param y:
    :param label_names: list of all available classes
    :param pred_proba:
    :param score
    :param filenames: also correspond to the index the image can be found in X
    :param filenames_train:
    :param filenames_test:
    :return:
    """

    print('>> Visualization')
    ### confusion matrices ###
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

    ### t-sne scatterplot ###
    if (pred_proba is not None):
        title = classifier + ', trained on ' + str(len(X_train)) + ' samples. Score: ' + str(score)
        X_embedded = plotting.t_sne_plot(X, X_test, y_test, y_pred, filenames_test, pred_proba, label_names, title, config.num_samples,
                                         classifier,
                                         "cupsnbottles", dims)

        ### image scatterplots ###
        X_all_embedded = tools.t_sne(X)
        indices_to_plot = None
        # image scatterplot misclassifications with frame depicting classification confidence
        inds_misclassification = np.argwhere(y_pred != y_test).flatten()
        if len(inds_misclassification) > 0:
            pass
            imgs = tools.load_images(config.path_dataset, filenames_test[inds_misclassification], filenames)
            title_imgs = str(len(imgs)) + ' test samples that were misclassified by ' + classifier
            plotting.image_conf_scatter(X_all_embedded, imgs, filenames_test[inds_misclassification], filenames, title_imgs, pred_proba[inds_misclassification], classifier)

        # image scatterplot ambiguous in test with frame denoting classification success
        if config.ambiguous_test_part > 0:
            indicesAmbiguous = np.array(df.loc[(df.ambiguous == 1) & (df.overlap == 0)]["index"])
            files_to_plot = np.intersect1d(indicesAmbiguous, filenames_test)
            imgs = tools.load_images(config.path_dataset, files_to_plot, filenames)
            title_imgs = str(len(imgs)) + ' ambiguous samples as classified by ' + classifier
            _, inds_in_test, _ = np.intersect1d(filenames_test, files_to_plot, return_indices=True)
            plotting.image_conf_scatter(X_all_embedded, imgs, files_to_plot, filenames, title_imgs, pred_proba[inds_in_test], classifier)

        # image scatterplot overlap in test with frame denoting classification success
        if config.overlap_test_part > 0:
            indicesOverlap = np.array(df.loc[(df.ambiguous == 0) & (df.overlap == 1)]["index"])
            files_to_plot = np.intersect1d(indicesOverlap, filenames_test)
            imgs = tools.load_images(config.path_dataset, files_to_plot, filenames)
            title_imgs = str(len(imgs)) + ' overlap samples as classified by ' + classifier
            _, inds_in_test, _ = np.intersect1d(filenames_test, files_to_plot, return_indices=True)
            plotting.image_conf_scatter(X_all_embedded, imgs, files_to_plot, filenames, title_imgs, pred_proba[inds_in_test], classifier)

        # image scatterplot low confidence (100 images by default)
        if pred_proba is not None:
            default_nb = 100
            if len(pred_proba) < default_nb:
                default_nb = len(pred_proba)
            pred_proba, filenames_test = (list(t) for t in zip(*sorted(zip(pred_proba, filenames_test))))
            imgs = tools.load_images(config.path_dataset, np.arange(default_nb), filenames_test)
            title_imgs = str(default_nb) + ' lowest confidence samples as classified by ' + classifier
            plotting.image_conf_scatter(X_all_embedded, imgs, filenames_test[:default_nb], filenames, title_imgs, pred_proba[:default_nb], classifier)
        print('>> DONE Visualization')


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
    print('>> Analysis')
    float_formatter = "{:.2f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})

    X_embedded = tools.t_sne(X)
    cluster_means = calculate_cluster_mean(X_embedded, y_train, label_names)

    filter_mask = create_filter_maske(y_test, y_pred, pred_proba, mode="all_test_samples") # possible modes: all_test_samples, wrong_test_samples, unconfident_test_samples
    true_labelnames = [label_names[i] for i in y_test[filter_mask]]
    predict_labelnames = [label_names[i] for i in y_pred[filter_mask].astype(int)]
    IDs = [indx_test[i] for i, value in enumerate(filter_mask) if value]

    ambiguous, overlaps = tools.read_properties()

    dataFalsePredict = {'IDs': IDs,
                        'True Label':y_test[filter_mask],
                        'True Labelname': true_labelnames,
                        'Predict Label':y_pred[filter_mask].astype(int),
                        'Predict Labelname': predict_labelnames
                        }

    if (len(ambiguous) != 0):
        ambiguous = ambiguous[IDs]
        dataFalsePredict['ambiguous'] = ambiguous

    if (len(overlaps) != 0):
        overlaps = overlaps[IDs]
        dataFalsePredict['overlap'] = overlaps

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
    print('>> DONE Analysis')


def main():
    X, y_encoded, y, label_names, df, filenames = tools.load_gt_data(config.num_samples, config.path_dataset)

    if config.normal_evaluation:
        X_train, X_test, y_train, y_test, filenames_train, filenames_test = model_selection.train_test_split(X,
                                                                                                             y_encoded,
                                                                                                             filenames,
                                                                                                             test_size=0.33,
                                                                                                             random_state=42)
    else:
        X_train, X_test, y_train, y_test, filenames_train, filenames_test = tools.adjust_dataset(X, y_encoded, filenames, df)

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

    analysis(X, y_encoded, X_test, y_test, y_pred, label_names, pred_proba, pred_proba_all, clf, filenames_test)
    #visualization(X, X_test, X_train, y_train, y_test, y_pred_train, y_pred, df, y, label_names, pred_proba, score, filenames, filenames_train, filenames_test)


if __name__ == "__main__":
    main()
