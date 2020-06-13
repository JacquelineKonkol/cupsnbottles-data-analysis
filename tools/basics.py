import cupsnbottles.load_cupsnbottles as load_cupsnbottles
import numpy as np
from sklearn import manifold
import os
import pickle as pkl
import pandas as pd
from PIL import Image

# preliminary, data format will change
# TODO num_samples noch einbauen, oder entfernen?
def load_gt_data(num_samples:int, path="cupsnbottles/" ):
    if path != 'cupsnbottles/':
        X = open_pkl(path, 'features.pkl')
        label_names = ['Can01', 'Can02', 'Can03', 'Coke001', 'Coke002Cup', 'WaterBottle']

        filenames = open_pkl(path, 'filenames.pkl')
        y_encoded = []
        y = []

        for i, filename in enumerate(filenames):
            filenames[i] = filename.replace('.png', '')
            for encoded, label in enumerate(label_names):
                if label in filename:
                    y.append(label)
                    y_encoded.append(encoded)

        df = {'index': np.arange(len(y)),
              'label': y}
        df = pd.DataFrame.from_dict(df)

    # to load the original cupsnbottles dataset
    else:
        X = load_cupsnbottles.load_features(path)
        print(X)
        df = load_cupsnbottles.load_properties(path)
        y = np.array(df.label)
        y_encoded = y.copy()
        label_names = np.unique(y)
        for (i, label) in enumerate(label_names):
            y_encoded[y == label] = i
        y_encoded = y_encoded.astype(int)

    if num_samples == 0:
        num_samples = len(X)
    label_names = np.unique(y[:num_samples])

    return X[:num_samples], y_encoded[:num_samples], y[:num_samples], label_names, df[:num_samples]

def open_pkl(path, file):
    with open(os.path.join(path,file), 'rb') as f:
        f = pkl.load(f)
    return f


def t_sne(X, dims=2, perplexity=30, learning_rate=200.0, n_iter=1000):
    """
    Calls t-SNE dimension reduction with default parameters. Can be adjusted.
    """
    tsne = manifold.TSNE(n_components=dims, init='random', perplexity=perplexity,
                         learning_rate = learning_rate,
                         n_iter=n_iter, n_iter_without_progress=300, method='barnes_hut',
                         random_state=0)
    return tsne.fit_transform(X)

def load_images(path, indices):
    """
    Loads images/ of dataset with any suffix (assumes images are titled with their indices)
    :param: path = path of the dataset, containing the images/ folder
    :param: indices = indices of the images to open, should correspond to the images name
    :returns: loaded images as a list
    """
    suffix = os.listdir(os.path.join(path, 'images'))[0].split(".")[-1]
    imgs = []
    for i in indices:
        imgs.append(Image.open(os.path.join(path, 'images', i + '.' + suffix)))
    return imgs
