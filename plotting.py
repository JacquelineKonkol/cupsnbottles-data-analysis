import cupsnbottles.load_cupsnbottles as load_cupsnbottles
import cupsnbottles.img_scatter as img_scatter

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

# TODO für jeden Classifier dafür sorgen, dass er die probabilities rausgibt (oh, auch in grid_search)

num_samples = 2179
dims = 2
#classifier = 'trained_classifiers/Nearest_Neighbors.joblib'
#classifier = 'trained_classifiers/Linear_SVM.joblib' # fix probability = False
#classifier = 'trained_classifiers/Nearest_Neighbors.joblib'
#classifier = 'trained_classifiers/Gaussian_Process.joblib'
#classifier = 'trained_classifiers/Random_Forest.joblib'
classifier = 'trained_classifiers/Neural_Net.joblib'

def t_sne(X, y_gt, y_pred, pred_proba, labels_old, fig_title, perplexity=30, learning_rate=200.0):
    ''' for the 2d and 3d case adds a visualization '''

    colors = np.array(["black", "grey", "orange", "darkred", "orchid",
                       "lime", "lightgrey", "red", "green", "#bcbd22", "c"])
    gt_colors = colors[y_gt[:num_samples]]
    pred_colors = colors[y_pred[:num_samples]]
    diff_colors = np.array(['darkgrey']*len(y_gt))
    diff_colors[pred_colors != gt_colors] = 'red'
    plotcolors = [gt_colors, pred_colors, diff_colors]

    tsne = manifold.TSNE(n_components=dims, init='random', perplexity=perplexity,
                         learning_rate = learning_rate,
                         n_iter=1000, n_iter_without_progress=300, method='barnes_hut',
                         random_state=0)
    X_embedded = tsne.fit_transform(X)

    if dims == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_co = np.expand_dims(X_embedded[:, 0], 1)
        y_co = np.expand_dims(X_embedded[:, 1], 1)
        z_co = np.expand_dims(X_embedded[:, 2], 1)
        ax.scatter(x_co, y_co, z_co, marker='.', color=gt_colors)
        for (i, label) in enumerate(labels_old):
            ax.scatter(np.array((1,1)), np.array((1,1)), np.array((1,1)),
                       marker='.', label=label, color=colors[i])
        plt.legend()
        plt.grid()
        plt.show()

        return X_embedded

    elif dims == 2:

        # add sample images into the plot
        imgs_to_plot = 200
        df = load_cupsnbottles.load_properties('')
        inds = np.array(df.index)
        random_inds = np.random.randint(0, len(y_gt), (imgs_to_plot))
        imgs = load_cupsnbottles.load_images('', inds[random_inds])
        artists = img_scatter.imageScatter(X_embedded[random_inds, 0],
                            X_embedded[random_inds, 1],imgs,img_scale=(13,13))
        plt.show()


        titles = ['Groundtruth', 'Predicted Labels', 'Difference', 'Confidence']
        fig, axes = plt.subplots(2, 2)
        fig.suptitle(fig_title)
        for i, ax in enumerate(axes.reshape(-1)):
            ax.set_title(titles[i])
            #ax.set_xticks([])
            #ax.set_yticks([])
            ax.grid()
            if i < 3:
                ax.scatter(X_embedded[:, 0], X_embedded[:, 1], marker='.', color=plotcolors[i])
            else:
                cmap = sns.cubehelix_palette(as_cmap=True)
                points = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=pred_proba, marker='.', cmap=cmap)
                fig.colorbar(points)


        for (i, label) in enumerate(labels_old):
            axes[0][1].scatter([], [], marker='.', label=label, color=colors[i])
        axes[0][1].legend(loc = 'upper right', bbox_to_anchor = (1.45, 1.2))
        plt.tight_layout()
        plt.show()

        return X_embedded
    else:
        return X_embedded

def load_gt_data():
    X = load_cupsnbottles.load_features('')
    df = load_cupsnbottles.load_properties('')
    y = np.array(df.label)
    labels_old = np.unique(y)
    for (i, label) in enumerate(labels_old):
        y[y == label] = i
    y = y.astype(int)
    X = X[:num_samples]
    y = y[:num_samples]

    return X, y, labels_old


X, y, labels_old = load_gt_data()
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.3)

# load the desired classifier
clf = load(str(classifier))

#clf = KNeighborsClassifier(**clf.best_params_)
#clf = SVC(**clf.best_params_)
#clf = GaussianProcessClassifier(**clf.best_params_)
#clf = DecisionTreeClassifier(**clf.best_params_)
#clf = RandomForestClassifier(**clf.best_params_)
clf = MLPClassifier(**clf.best_params_)
#clf = GaussianNB(**clf.best_params_)
#clf = QuadraticDiscriminantAnalysis(**clf.best_params_)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
for (i, label) in enumerate(labels_old):
    y_pred[y_pred == label] = i
y_pred = y_pred.astype(int)
pred_proba = clf.predict_proba(X_test)
pred_proba = np.max(pred_proba, axis=1)

#title = "Nearest Neighbors, trained on " + str(len(X_train)) + ' samples. Score: ' + str(score)
#title = "Linear SVM, trained on " + str(len(X_train)) + ' samples. Score: ' + str(score)
#title = "RBF SVM, trained on " + str(len(X_train)) + ' samples. Score: ' + str(score)
#title = "Gaussian Process, trained on " + str(len(X_train)) + ' samples. Score: ' + str(score)
#title = "Decision Tree, trained on " + str(len(X_train)) + ' samples. Score: ' + str(score)
#title = "Random Forest, trained on " + str(len(X_train)) + ' samples. Score: ' + str(score)
title = "Neural Net, trained on " + str(len(X_train)) + ' samples. Score: ' + str(score)
#title = "Naive Bayes, trained on " + str(len(X_train)) + ' samples. Score: ' + str(score)
#title = "QDA, trained on " + str(len(X_train)) + ' samples. Score: ' + str(score)


_ = t_sne(X_test, y_test, y_pred, pred_proba, labels_old, title)
