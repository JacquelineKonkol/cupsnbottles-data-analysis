import cupsnbottles.load_cupsnbottles as load_cupsnbottles
import cupsnbottles.img_scatter as img_scatter
import tools.basics as tools
import tools.settings as settings

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
from matplotlib.collections import LineCollection
import matplotlib.image as mpimg
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

config = settings.config()
classifier = config.classifier

def plot_confusion_matrix(cm, classes,
                          img_name,
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
    plt.savefig("./plots/conf_matrix_" + classifier + img_name)
    plt.show()



def t_sne_plot(X, X_test, y_gt, y_pred, filenames_test, pred_proba, labels_old, fig_title, num_samples, classifier, img_name, dims=2, perplexity=30, learning_rate=200.0):
    """
    nD case: returns data embedded into n dimensions using t_sne
    3D case: simple t-SNE 3D plot with gt labels
    2D case: t-SNE plot with gt labels, predicted labels, the difference between
             the two and the classification confidence for each prediction
             the embedding is calculated with t-SNE on all data points
    :returns: embedded data in n-dim
    """
    if num_samples == 0: num_samples = len(X_test)
    print(num_samples)
    colors = np.array(["black", "grey", "orange", "darkred", "orchid",
                       "lime", "lightgrey", "red", "green", "#bcbd22", "c"])
    gt_colors = colors[y_gt]
    pred_colors = colors[y_pred]
    diff_colors = np.array(['darkgrey']*len(y_gt))
    diff_colors[pred_colors != gt_colors] = 'red'
    plotcolors = [gt_colors, pred_colors, diff_colors]

    X_embedded_all = tools.t_sne(X)
    X_embedded = tools.t_sne(X_test)

    if dims == 3:
        # right now visualizes only gt
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
        plt.savefig("./plots/t-sne_3dim_" + img_name)
        plt.show()

        return X_embedded

    elif dims == 2:
        titles = ['Groundtruth', 'Predicted Labels', 'Difference', 'Confidence']
        fig, axes = plt.subplots(2, 2, figsize=(15,10))
        fig.suptitle(fig_title, fontsize=20)

        for i, ax in enumerate(axes.reshape(-1)):
            ax.set_title(titles[i])
            #ax.set_xticks([])
            #ax.set_yticks([])
            ax.grid()
            if i < 3:
                ax.scatter(X_embedded_all[filenames_test, 0], X_embedded_all[filenames_test, 1], marker='.', color=plotcolors[i])
                #ax.scatter(X_embedded[:, 0], X_embedded[:, 1], marker='.', color=plotcolors[i]) # here t-sne was performed only on test data
            else:
                cmap = sns.cubehelix_palette(as_cmap=True)
                points = ax.scatter(X_embedded_all[filenames_test, 0], X_embedded_all[filenames_test, 1], c=pred_proba, marker='.', cmap=cmap)
                #points = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], c=pred_proba, marker='.', cmap=cmap) # here t-sne was performed only on test data
                fig.colorbar(points)


        for (i, label) in enumerate(labels_old):
            axes[0][1].scatter([], [], marker='.', label=label, color=colors[i])
        axes[0][1].legend(loc = 'upper right', bbox_to_anchor = (1.45, 1.2))
        plt.tight_layout()

        fig.savefig('plots/' + img_name + classifier.replace(' ', '_') + '.png', bbox_inches='tight')
        plt.show()

        return X_embedded
    else:
        return X_embedded


def image_conf_scatter(X_all_embedded, imgs, files_to_plot, filenames, title, pred_proba, plotting_type):
    """
    Creates a scatterplot containing selected image samples. These are framed with a color representing
    the prediction confidence from the classifier.
    :param: X_all_embedded = should be a 2D embedding of the whole dataset
    :param: imgs = images to include into the scatterplot
    :param: files_to_plot = filenames of images in question
    :param: filenames = list of all filenames in the dataset
    :param: title = title of the figure
    :param: pred_proba = classification probability values for the samples in question
    """

    # find where the images for plotting are amongst all the available samples in the dataset
    _, inds_relativeToAll, _ = np.intersect1d(filenames, files_to_plot, return_indices=True)

    fig, axs = plt.subplots(2, figsize=(15, 10), gridspec_kw={'height_ratios': [15,1]})
    fig.subplots_adjust(bottom=0.5)
    cmap = matplotlib.cm.cool
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cb1 = matplotlib.colorbar.ColorbarBase(axs[1], cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Prediction Probability')

    # frame images according to their prediction probability
    imgs_framed = []
    for i, img in enumerate(imgs):
        img = np.asarray(img)
        col = cmap(pred_proba[i])
        imgs_framed.append(img_scatter.frameImage(img,col))


    artists = img_scatter.imageScatter(X_all_embedded[inds_relativeToAll, 0],
                                       X_all_embedded[inds_relativeToAll, 1],imgs_framed, ax=axs[0], img_scale=(30,30))
    fig.suptitle(title, fontsize=20)
    plt.grid()

    fig.savefig('plots/' + classifier.replace(' ', '_') + plotting_type + '.png')
    plt.show()
