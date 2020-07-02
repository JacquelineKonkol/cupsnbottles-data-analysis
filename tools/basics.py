import cupsnbottles.load_cupsnbottles as load_cupsnbottles
import numpy as np
from sklearn import manifold
import os
import pickle as pkl
import pandas as pd
from PIL import Image
from tools.settings import config

config = config()

# TODO num_samples noch einbauen, oder entfernen?
def load_gt_data(num_samples=config.num_samples, path=config.path_dataset):
    if path == 'dataset01/':
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

        y = np.array(y)
        y_encoded = np.array(y_encoded)
        df = {'index': np.arange(len(y)),
              'label': y}
        df = pd.DataFrame.from_dict(df)

    # to load the original cupsnbottles dataset
    elif path == "cupsnbottles/":
         X = load_cupsnbottles.load_features(path)
         print(X)
         df = load_cupsnbottles.load_properties(path)
         y = np.array(df.label)
         y_encoded = y.copy()
         label_names = np.unique(y)
         for (i, label) in enumerate(label_names):
             y_encoded[y == label] = i
         y_encoded = y_encoded.astype(int)

    elif path == "dataset02/":
        X = open_pkl(path, 'features.pkl')
        properties = csv_to_df(path, 'properties.csv')
        y = np.array(properties.object_class)
        label_names = np.unique(y)
        y_encoded = y.copy()
        for (i, label) in enumerate(label_names):
            y_encoded[y == label] = i
        y_encoded = y_encoded.astype(int)
        filenames = properties.index
        df = properties

    if num_samples == 0:
        num_samples = len(X)
    #label_names = np.unique(y[:num_samples])
    #print(label_names)

    return X[:num_samples], y_encoded[:num_samples], y[:num_samples], label_names, df[:num_samples], filenames[:num_samples]

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

def load_images(path, indices, filenames):
    """
    Loads images/ of dataset with any suffix
    :param: path = path of the dataset, containing the images/ folder
    :param: indices = indices of the images to open
    :returns: loaded images as a list
    """
    suffix = os.listdir(os.path.join(path, 'images'))[0].split(".")[-1]
    imgs = []
    for i in indices:
        imgs.append(Image.open(os.path.join(path, 'images', str(filenames[i]) + '.' + str(suffix))))
    return imgs

def csv_to_df(path, file):
    return pd.read_csv(os.path.join(path,file))

def categorize_data(df):
    indicesAmbiguous = df.loc[df.ambiguous == 1]
    indicesOverlap = df.loc[df.overlap == 1]
    indicesVanilla = df.loc[(df.ambiguous == 0) & (df.overlap == 0)]
    indicesBoth = df.loc[(df.ambiguous == 1) & (df.overlap == 1)]
    return indicesVanilla, indicesOverlap, indicesAmbiguous, indicesBoth

def adjust_dataset(X, filenames, df, ambiguous_perc_train, overlap_perc_train, ambiguous_perc_test, overlap_perc_test):
    indicesVanilla, indicesOverlap, indicesAmbiguous, indicesBoth = categorize_data(df)
    nb_train = int(2/3 * len(X))
    nb_test = int(1/3 * len(X))
    nb_samples_ambiguous_train = int(ambiguous_perc_train/100 * nb_train)
    nb_samples_overlap_train = int(overlap_perc_train/100 * nb_train)
    nb_samples_ambiguous_test = int(ambiguous_perc_test/100 * nb_test)
    nb_samples_overlap_test = int(overlap_perc_test/100 * nb_test)
    nb_samples_vanilla_train = nb_train - nb_samples_ambiguous_train - nb_samples_overlap_train
    nb_samples_vanilla_test = nb_test - nb_samples_ambiguous_test - nb_samples_overlap_test

    np.random.choice(indicesAmbiguous, nb_samples_ambiguous_train)
    np.random.choice(indicesOverlap, nb_samples_overlap_train)

    #X_train, X_test, y_train, y_test, indx_train, indx_test= model_selection.train_test_split(X, y_encoded, indx, test_size=0.33, random_state=42)
    pass
