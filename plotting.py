# TODO Aufruf der einzelnen classifier aufhübschen
# TODO für jeden Classifier dafür sorgen, dass er die probabilities rausgibt
# TODO save fig
# TODO include confusion matrix
# TODO confidence Balken zwischen 0 und 1 skalieren

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

num_samples = 2179
dims = 2
classifier = 'Neural Net'

path_trained_classifiers = 'trained_classifiers/'

def t_sne_plot(X, y_gt, y_pred, pred_proba, labels_old, fig_title, perplexity=30, learning_rate=200.0):
    """
    nD case: returns data embedded into n dimensions using t_sne
    3D case: simple t-SNE 3D plot with gt labels
    2D case: t-SNE plot with gt labels, with predicted labels, the difference between
             the two and the classification confidence for each prediction
    :returns: embedded data in n-dim
    """

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

# -- should be removed if designated function is working

        # add sample images into the plot
        imgs_to_plot = 200
        df = load_cupsnbottles.load_properties('cupsnbottles/')
        inds = np.array(df.index)
        random_inds = np.random.randint(0, len(y_gt), (imgs_to_plot))
        imgs = load_cupsnbottles.load_images('cupsnbottles/', inds[random_inds])
        artists = img_scatter.imageScatter(X_embedded[random_inds, 0],
                            X_embedded[random_inds, 1],imgs,img_scale=(13,13))
        plt.show()
# -- should be removed if designated function is working

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


########### TODO testen
def confidence_scatter(X_embedded, df, imgs, indices, title):
    """
    :param: X_embedded = should be 2D
    :param: df = dataframe containing the images load_properties
    :param: imgs = images to include into the scatterplot
    :param: indices = of the images to include
    """
    fig = plt.figure()
    artists = img_scatter.imageScatter(X_embedded[indices, 0],
                         X_embedded[indices, 1],imgs,img_scale=(13,13))
    fig.suptitle(title)
    plt.grid()
    plt.show()

    # add sample images into the scatter plot
    # imgs_to_plot = 200
    ## df = load_cupsnbottles.load_properties('cupsnbottles/')
    ## inds = np.array(df.index)
    ## random_inds = np.random.randint(0, len(y_gt), (imgs_to_plot))
    ## imgs = load_cupsnbottles.load_images('cupsnbottles/', inds[random_inds])
    # artists = img_scatter.imageScatter(X_embedded[random_inds, 0],
    #                     X_embedded[random_inds, 1],imgs,img_scale=(13,13))
    # plt.show()
    pass

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

    return X, y, labels_old


X, y, labels_old = load_gt_data()
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

classifiers = [
     KNeighborsClassifier(**clf.best_params_),
     SVC(**clf.best_params_),
     SVC(**clf.best_params_),
     GaussianProcessClassifier(**clf.best_params_),
     DecisionTreeClassifier(**clf.best_params_),
     RandomForestClassifier(**clf.best_params_),
     MLPClassifier(**clf.best_params_),
     GaussianNB(**clf.best_params_),
     QuadraticDiscriminantAnalysis(**clf.best_params_)]

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)
for (i, label) in enumerate(labels_old):
    y_pred[y_pred == label] = i
y_pred = y_pred.astype(int)
pred_proba = clf.predict_proba(X_test)
pred_proba = np.max(pred_proba, axis=1)

title = classifier + ', trained on " + str(len(X_train)) + ' samples. Score: ' + str(score)


X_embedded = t_sne_plot(X_test, y_test, y_pred, pred_proba, labels_old, title)
