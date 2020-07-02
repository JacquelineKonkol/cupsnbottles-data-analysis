# TODO generalize so different datasets can be used
# TODO falls vortrainierter Classifier: welche Testdaten sollen genommen werden?
# TODO Option, neu trainierten Classifier zu speichern?

import cupsnbottles.load_cupsnbottles as load_cupsnbottles
import tools.basics as tools
import plotting
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn import manifold, model_selection
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd

################################################################################
####################################specify#####################################

classifier = "Nearest Neighbors" # look up in classifier_names list
use_pretrained_classifier = False
imgs_falsely_classified = False # only misclassified images are used in
                               # the scatterplot, random otherwise
all_samples = 0 # 0 is the default to load the whole dataset
num_samples = all_samples
dims = 2

path_dataset = "dataset02/" # TODO generalize so different datasets can be used
path_trained_classifiers = 'trained_classifiers/' # keep in mind that we don't want to test on data the model was trained on
path_best_params = 'classifiers_best_params/'

################################################################################

classifier_names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                    "Decision Tree", "Random Forest", "Neural Net", "Naive Bayes", "QDA"]

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


def prepare_clf(X_train, y_train):
    if use_pretrained_classifier:
        # load the desired trained classifier
        clf = load(path_trained_classifiers + path_dataset + classifier.replace(' ', '_') + ".joblib")

    else:
        # train anew with best model parameters
        loaded_params = load(path_best_params + path_dataset + classifier.replace(' ', '_') + "_params.joblib")
        clf = classifiers[classifier_names.index(classifier)]
        clf.set_params(**loaded_params)
        clf.fit(X_train, y_train)
    return clf


def visualization(X_test, X_train, y_train, y_test, y_pred_train, y_pred, df, y, label_names, pred_proba, score, filenames):
    # plotting t-SNE
    title = classifier + ', trained on ' + str(len(X_train)) + ' samples. Score: ' + str(score)
    X_embedded = plotting.t_sne_plot(X_test, y_test, y_pred, pred_proba, label_names, title, num_samples, classifier,
                                     "cupsnbottles", dims)
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

    imgs = tools.load_images(path_dataset, inds[indices], filenames)
    plotting.image_conf_scatter(X_embedded, imgs, indices, title_imgs, pred_proba, classifier)


#TODO umstrukturieren
def calculate_cluster_mean(X_embedded, y_test, label_names):
    cluster_infos = {}
    for i in range(len(label_names)):
        X_class_indx =  y_test == i
        X_selected = X_embedded[X_class_indx]
        mean = np.mean(X_selected, axis=0)
        var = np.mean(X_selected, axis=0)
        cluster_infos[i] = {'mean':mean, "var":var}

    return cluster_infos



#TODO umstrukturieren
def analysis(X_test, X_train, y_train, y_test, y_pred_train, y_pred, df, y, label_names, pred_proba, score):
    X_embedded = tools.t_sne(X_test) # nur mit X_test? Alle sinnvoller für Cluster mean?
    cluster_infos = calculate_cluster_mean(X_embedded, y_test, label_names)
    print(cluster_infos) #in dataFalsePredict Abstände zu Klassen speichern
    y_pred[0] = 1 #only for testing if y_test has no false classified samples
    falsePredict = y_test != y_pred
    dataFalsePredict = {'True Label':y_test[falsePredict],
                        'Predict Label':y_pred[falsePredict],
                        'Predict Prob.':pred_proba[falsePredict]
                        }
    df = pd.DataFrame(dataFalsePredict, columns=['True Label', 'Predict Label', 'Predict Prob.']) # Todo Index, Id der Punkte mitfesthalten zur Identifikation, Todo labelname hinzufügen
    print(df) # Todo speichern als csv wie in gridsearch

def main():
    X, y_encoded, y, label_names, df, filenames = tools.load_gt_data(num_samples, path_dataset)
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

    filenames_train = np.array(filenames_train).tolist()
    filenames_test = np.array(filenames_test).tolist()

    for i, row in df.iterrows():
        if row['ambiguous'] == 1:
            print('count')
            if row['index'] in filenames_train:
                X_train_only_ambiguous.append(X_train[filenames_train.index(row['index'])])
                y_train_only_ambiguous.append(y_train[filenames_train.index(row['index'])])
            else:
                X_test_only_ambiguous.append(X_test[filenames_test.index(row['index'])])
                y_test_only_ambiguous.append(y_test[filenames_test.index(row['index'])])
        elif row['ambiguous'] == 0:
            if row['index'] in filenames_train:
                X_train_without_ambiguous.append(X_train[filenames_train.index(row['index'])])
                y_train_without_ambiguous.append(y_train[filenames_train.index(row['index'])])
            else:
                X_test_without_ambiguous.append(X_test[filenames_test.index(row['index'])])
                y_test_without_ambiguous.append(y_test[filenames_test.index(row['index'])])


    if training == 'ambiguous only':
        X_train = X_train_only_ambiguous
        y_train = y_train_only_ambiguous
    elif training == 'without ambiguous':
        X_train = X_train_without_ambiguous
        y_train = y_train_without_ambiguous
    if testing == 'ambiguous only':
        X_test = X_test_only_ambiguous
        y_test = y_test_only_ambiguous
    elif testing == 'without ambiguous':
        X_test = X_test_without_ambiguous
        y_test = y_test_without_ambiguous
    ### TEMP

    print(y_train)
    print(y_test)

    clf = prepare_clf(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    score = clf.score(X_test, y_test)
    pred_proba = clf.predict_proba(X_test)
    pred_proba = np.max(pred_proba, axis=1)

    analysis(X_test, X_train, y_train, y_test, y_pred_train, y_pred, df, y, label_names, pred_proba, score)
    visualization(X_test, X_train, y_train, y_test, y_pred_train, y_pred, df, y, label_names, pred_proba, score, filenames_test)


if __name__ == "__main__":
    main()
