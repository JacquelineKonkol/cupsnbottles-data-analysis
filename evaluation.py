# TODO generalize so different datasets can be used
# TODO falls vortrainierter Classifier: welche Testdaten sollen genommen werden?
# TODO confusion Matrix verwenden
# TODO img_scatter Sache? Vielleicht diejenigen Puntke, die falsch oder mit wenig confidence klassifiziert wurden darstellen?

import cupsnbottles.load_cupsnbottles as load_cupsnbottles
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

################################################################################
####################################specify#####################################

classifier = "RBF SVM" # look up in classifier_names list
use_pretrained_classifier = False
imgs_falsely_classified = True # uses a confidence threshold to decide what images to
                              # use in the scatterplot, random otherwise

num_samples = 500
dims = 2


path_dataset = '' # TODO generalize so different datasets can be used
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


def load_gt_data():
    X = load_cupsnbottles.load_features('cupsnbottles/')
    df = load_cupsnbottles.load_properties('cupsnbottles/')
    y = np.array(df.label)
    labels_old = np.unique(y)
    for (i, label) in enumerate(labels_old):
        y[y == label] = i
    y = y.astype(int)
    X = X[:num_samples]
    y = y[:num_samples]

    return X, y, labels_old, df


def main():


    X, y, labels_old, df = load_gt_data()
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.3)

    if use_pretrained_classifier:
        # load the desired trained classifier
        clf = load(path_trained_classifiers + classifier.replace(' ', '_') + ".joblib")
    else:
        # train anew with best model parameters
        loaded_params = load(path_best_params + classifier.replace(' ', '_') + "_params.joblib")
        clf = classifiers[classifier_names.index(classifier)]
        clf.set_params(**loaded_params)
        clf.fit(X_train, y_train)


    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    for (i, label) in enumerate(labels_old):
        y_pred[y_pred == label] = i
    y_pred = y_pred.astype(int)
    pred_proba = clf.predict_proba(X_test)
    pred_proba = np.max(pred_proba, axis=1)

    # plotting t-SNE
    title = classifier + ', trained on ' + str(len(X_train)) + ' samples. Score: ' + str(score)
    X_embedded = plotting.t_sne_plot(X_test, y_test, y_pred, pred_proba, labels_old, title, num_samples, dims)

    inds = np.array(df.index)
    indices = None

    if imgs_falsely_classified:
        indices = np.argwhere(y_pred != y_test).flatten()
        print(indices)

        title_imgs = 'Images that were falsely classified by ' + classifier
    # plot random images into the scatter
    else:
        imgs_to_plot = 20
        indices = np.random.randint(0, len(y_test), (imgs_to_plot))
        title_imgs = str(imgs_to_plot) + ' random images'

    imgs = load_cupsnbottles.load_images('cupsnbottles/', inds[indices])
    plotting.image_conf_scatter(X_embedded, imgs, indices, title_imgs, pred_proba)

if __name__ == "__main__":
    main()
