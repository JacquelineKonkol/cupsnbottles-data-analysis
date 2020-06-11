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

################################################################################
####################################specify#####################################

classifier = "Decision Tree" # look up in classifier_names list
use_pretrained_classifier = False
imgs_falsely_classified = False # only misclassified images are used in
                               # the scatterplot, random otherwise

num_samples = 50
dims = 2

path_dataset = "test_dataset/" # TODO generalize so different datasets can be used
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
        clf = load(path_trained_classifiers + classifier.replace(' ', '_') + ".joblib")
    else:
        # train anew with best model parameters
        loaded_params = load(path_best_params + classifier.replace(' ', '_') + "_params.joblib")
        clf = classifiers[classifier_names.index(classifier)]
        clf.set_params(**loaded_params)
        clf.fit(X_train, y_train)
    return clf


def visualization(X_test, X_train, df, label_names, pred_proba, score, y_pred, y_test, y_train, y_pred_train):
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

    cm_train = metrics.confusion_matrix(y_train, y_pred_train)
    plotting.plot_confusion_matrix(cm_train, classes=label_names, img_name="absolute_cupsnbottles_train", cmap=plt.cm.Blues)
    plotting.plot_confusion_matrix(cm_train, classes=label_names, img_name="norm_cupsnbottles_train", normalize=True,
                                   title='Normalized confusion matrix, trainings data', cmap=plt.cm.Blues)
    cm = metrics.confusion_matrix(y_test, y_pred)
    plotting.plot_confusion_matrix(cm, classes=label_names, img_name="absolute_cupsnbottles", cmap=plt.cm.Greens)
    plotting.plot_confusion_matrix(cm, classes=label_names, img_name="norm_cupsnbottles", normalize=True,
                                   title='Normalized confusion matrix', cmap=plt.cm.Greens)

    imgs = tools.load_images(path_dataset, inds[indices])
    plotting.image_conf_scatter(X_embedded, imgs, indices, title_imgs, pred_proba, classifier)


def main():

    X, y_encoded, y, label_names, df = tools.load_gt_data(num_samples, path_dataset)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y_encoded, test_size=0.33, random_state=42)
    clf = prepare_clf(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    score = clf.score(X_test, y_test)
    y_pred = y_pred.astype(int)
    y_pred_train = y_pred_train.astype(int)
    pred_proba = clf.predict_proba(X_test)
    pred_proba = np.max(pred_proba, axis=1)

    visualization(X_test, X_train, df, label_names, pred_proba, score, y_pred, y_test, y_train, y_pred_train)


if __name__ == "__main__":
    main()
