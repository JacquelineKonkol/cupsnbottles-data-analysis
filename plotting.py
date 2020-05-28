# TODO Aufruf der einzelnen classifier aufhübschen
# TODO für jeden Classifier dafür sorgen, dass er die probabilities rausgibt
# TODO save fig
# TODO include confusion matrix
# TODO confidence Balken zwischen 0 und 1 skalieren

import cupsnbottles.load_cupsnbottles as load_cupsnbottles
import cupsnbottles.img_scatter as img_scatter

import itertools
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
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.rcParams["font.family"] = 'DejaVu Sans'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def t_sne_plot(X, y_gt, y_pred, pred_proba, labels_old, fig_title, num_samples, dims=2, perplexity=30, learning_rate=200.0):
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
        fig.suptitle(fig_title,  y=0.99)
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
def image_scatter(X_embedded, df, imgs, indices, title):
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
