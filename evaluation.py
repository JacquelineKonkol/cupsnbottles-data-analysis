# TODO generalize so different datasets can be used
# TODO falls vortrainierter Classifier: welche Testdaten sollen genommen werden?
# TODO falls Classifier beste Parameter laden und mit diesen neu trainiert werden soll
# TODO img_scatter Sache? Vielleicht diejenigen Puntke, die falsch oder mit wenig confidence klassifiziert wurden darstellen?

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

"""specify"""
classifier = 'Neural Net'
use_pretrained_classifier = True
imgs_scatter_threshold = True # uses a confidence threshold to decide what images to
                              # use in the scatterplot, random otherwise

num_samples = 2179
dims = 2


path_dataset = '' # TODO generalize so different datasets can be used
path_trained_classifiers = 'trained_classifiers/' # keep in mind that we don't want to test on data the model was trained on
path_best_params = 'classifiers_best_params/'

def prepare_clf():
    if use_pretrained_classifier:
        # load the desired trained classifier
        clf = load(path_trained_classifiers + classifier.replace(' ', '_') + ".joblib")
    else:
        pass
        ## TODO load with best_params first
        # something like that
        # classifiers = [
        #      KNeighborsClassifier(**clf.best_params_),
        #      SVC(**clf.best_params_),
        #      SVC(**clf.best_params_),
        #      GaussianProcessClassifier(**clf.best_params_),
        #      DecisionTreeClassifier(**clf.best_params_),
        #      RandomForestClassifier(**clf.best_params_),
        #      MLPClassifier(**clf.best_params_),
        #      GaussianNB(**clf.best_params_),
        #      QuadraticDiscriminantAnalysis(**clf.best_params_)]

        #clf.fit(X_train, y_train)

    return clf


def main():

    X, y_encoded, y, label_names, df = tools.load_gt_data(num_samples)
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y_encoded, test_size=0.33, random_state=42)

    clf = prepare_clf()

    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    for (i, label) in enumerate(label_names):
        y_pred[y_pred == label] = i
    y_pred = y_pred.astype(int)
    pred_proba = clf.predict_proba(X_test)
    pred_proba = np.max(pred_proba, axis=1)

    title = classifier + ', trained on ' + str(len(X_train)) + ' samples. Score: ' + str(score)


    X_embedded = plotting.t_sne_plot(X_test, y_test, y_pred, pred_proba, label_names, title, num_samples,"cupsnbottles", dims)

    cm = metrics.confusion_matrix(y_test, y_pred)
    plotting.plot_confusion_matrix(cm,classes=label_names, img_name="absolute_cupsnbottles", cmap=plt.cm.Greens)
    plotting.plot_confusion_matrix(cm, classes=label_names, img_name="norm_cupsnbottles", normalize=True, title='Normalized confusion matrix', cmap=plt.cm.Greens)

    # TODO with imgs_scatter_threshold
    # indices sind entweder random oder könnten zB den Datenpunkten entsprechen,
    # die am unsichersten klassifiziert wurden
    conf_threshold = 0.7
    #imgs = load_cupsnbottles.load_images('cupsnbottles/', inds[random_inds])
    #title_imgs = "Images that were classified with a confidence below " + str(conf_threshold)
    #plotting.image_scatter(X_embedded, df, indices, title_imgs)


if __name__ == "__main__":
    main()
