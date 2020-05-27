# TODO generalize so different datasets can be used
# TODO falls trainierter Classifier genommen werden soll
# TODO falls Classifier beste Parameter laden und mit diesen neu trainiert werden soll
# TODO plotting.py verwenden
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

"""specify"""
classifier = None #
classifier = 'Neural Net'

num_samples = 2179
dims = 2


path_dataset = '' # TODO generalize so different datasets can be used
path_trained_classifiers = 'trained_classifiers/' # keep in mind that we don't want to test on data the model was trained on
path_best_params = 'classifiers_best_params/'

def load_gt_data():
    X = load_cupsnbottles.load_features('cupsnbottles/')
    print(X)
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

# das ist noch alles scheisse, weil der die trainierten Classifier lädt und daraus die parameter nimmt

    X, y, labels_old, df = load_gt_data()
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.3)

    # load the desired classifier
    clf = load(path_trained_classifiers + classifier.replace(' ', '_') + ".joblib")

    #clf = KNeighborsClassifier(**clf.best_params_)
    #clf = SVC(**clf.best_params_)
    #clf = GaussianProcessClassifier(**clf.best_params_)
    #clf = DecisionTreeClassifier(**clf.best_params_)
    #clf = RandomForestClassifier(**clf.best_params_)
    clf = MLPClassifier(**clf.best_params_)
    #clf = GaussianNB(**clf.best_params_)
    #clf = QuadraticDiscriminantAnalysis(**clf.best_params_)

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

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    for (i, label) in enumerate(labels_old):
        y_pred[y_pred == label] = i
    y_pred = y_pred.astype(int)
    pred_proba = clf.predict_proba(X_test)
    pred_proba = np.max(pred_proba, axis=1)

    title = classifier + ', trained on " + str(len(X_train)) + ' samples. Score: ' + str(score)


    X_embedded = plotting.t_sne_plot(X_test, y_test, y_pred, pred_proba, labels_old, title)
    # TODO, also in plotting
    # indices sind entweder random oder könnten zB den Datenpunkten entsprechen,
    # die am unsichersten klassifiziert wurden
    # plotting.confidence_scatter(X_embedded, df, indices, title_imgs)

if __name__ == "__main__":
    main()
