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
    # in dataset01/ the properties.csv does not suffice for all the function later used
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
         df = load_cupsnbottles.load_properties(path)
         y = np.array(df.label)
         y_encoded = y.copy()
         label_names = np.unique(y)
         for (i, label) in enumerate(label_names):
             y_encoded[y == label] = i
         y_encoded = y_encoded.astype(int)

    # using the norm that the image-files are named by their index and the properties.csv
    # provides information on ambiguity and overlaps
    else:
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
    :param: filenames = should be the same order as indices
    :returns: loaded images as a list
    """
    suffix = os.listdir(os.path.join(path, 'images'))[0].split(".")[-1]
    imgs = []
    for i in indices:
        imgs.append(np.array(Image.open(os.path.join(path, 'images', str(filenames[i]) + '.' + str(suffix)))))
    return imgs

def csv_to_df(path, file):
    return pd.read_csv(os.path.join(path,file))


def adjust_dataset(X, y_encoded, filenames, df):
    """
    Considers the data categories vanilla, ambiguous, overlap and both (i.e. ambiguous and overlap)
    and uses information given in config [DATASET] to split training and testing data according to
    the desired propotions of each category
    :param: X = raw data matrix
    :param: y_encoded = labels of x as integers
    :param: filenames = also correspond to the index the image can be found in X
    :param: df = pandas dataframe of the meta-data given in properties.csv
    :returns:
    """

    ### TODO soll der Fall both anders gehändelt werden?

    # determine sample categories
    indicesAmbiguous = np.array(df.loc[(df.ambiguous == 1) & (df.overlap == 0)]["index"])
    indicesOverlap = np.array(df.loc[(df.ambiguous == 0) & (df.overlap == 1)]["index"])
    indicesVanilla = np.array(df.loc[(df.ambiguous == 0) & (df.overlap == 0)]["index"])
    indicesBoth = np.array(df.loc[(df.ambiguous == 1) & (df.overlap == 1)]["index"])

    maskAmbiguous = np.array((df['ambiguous'] == 1) & (df['overlap'] == 0))
    maskOverlap = np.array((df['ambiguous'] == 0) & (df['overlap'] == 1))
    maskVanilla = np.array((df['ambiguous'] == 0) & (df['overlap'] == 0))
    maskBoth = np.array((df['ambiguous'] == 1) & (df['overlap'] == 1))

    # shuffle dataset
    print('>> Preparing Dataset')
    print('Total available samples: ', len(X))
    shuffler = np.random.permutation(len(X))
    X, y_encoded, filenames = X[shuffler], y_encoded[shuffler], filenames[shuffler]
    maskVanilla, maskAmbiguous, maskOverlap, maskBoth = maskVanilla[shuffler], maskAmbiguous[shuffler], maskOverlap[shuffler], maskBoth[shuffler]

    # split each category into train and test according to requested proportion
    X_train, y_train, filenames_train = [], [], []
    X_test, y_test, filenames_test = [], [], []
    categories = ['vanilla', 'ambiguous', 'overlap', 'both']
    lengths = [len(indicesVanilla), len(indicesAmbiguous), len(indicesOverlap), len(indicesBoth)]
    X_sets = [X[maskVanilla], X[maskAmbiguous], X[maskOverlap], X[maskBoth]]
    y_sets = [y_encoded[maskVanilla], y_encoded[maskAmbiguous], y_encoded[maskOverlap], y_encoded[maskBoth]]
    filenames_sets = [filenames[maskVanilla], filenames[maskAmbiguous], filenames[maskOverlap], filenames[maskBoth]]
    train_parts = [config.vanilla_train_part, config.ambiguous_train_part, config.overlap_train_part, config.both_train_part]
    test_parts = [config.vanilla_test_part, config.ambiguous_test_part, config.overlap_test_part, config.both_test_part]

    # for both training and testing
    for i in range(2):
        # add every category according to requested proportion
        for category, _ in enumerate(X_sets):
            # edge case where the data can't be split in equal halves
            if train_parts[category] and test_parts[category] and len(X_sets[category]) % 2 == 1 and i == 0:
                print('Using ' + str(int(round(train_parts[category] * lengths[category]))) + ' samples from ' + categories[category] + " in training.")
                X_train.extend(X_sets[category][:int(round(train_parts[category] * lengths[category]))])
                y_train.extend(y_sets[category][:int(round(train_parts[category] * lengths[category]))])
                filenames_train.extend(filenames_sets[category][:int(round(train_parts[category] * lengths[category]))])
                print('Using ' + str(int(round(test_parts[category] * lengths[category]))-1) + ' samples from ' +categories[category] + " in testing.")
                X_test.extend(X_sets[category][-int(round(test_parts[category] * lengths[category]))-1:])
                y_test.extend(y_sets[category][-int(round(test_parts[category] * lengths[category]))-1:])
                filenames_test.extend(filenames_sets[category][-int(round(test_parts[category] * lengths[category]))-1:])
            else:
                # add training part of category to training variables
                if i == 0:
                    print('Using ' + str(int(round(train_parts[category] * lengths[category]))) + ' samples from ' + categories[category] + " in training.")
                    if train_parts[category] != 0:
                        X_train.extend(X_sets[category][:int(round(train_parts[category] * lengths[category]))])
                        y_train.extend(y_sets[category][:int(round(train_parts[category] * lengths[category]))])
                        filenames_train.extend(filenames_sets[category][:int(round(train_parts[category] * lengths[category]))])
                # add testing part of category to testing variables
                else:
                    print('Using ' + str(int(round(test_parts[category] * lengths[category]))) + ' samples from ' + categories[category] + " in testing.")
                    if test_parts[category] != 0:
                        X_test.extend(X_sets[category][-int(round(test_parts[category] * lengths[category])):])
                        y_test.extend(y_sets[category][-int(round(test_parts[category] * lengths[category])):])
                        filenames_test.extend(filenames_sets[category][-int(round(test_parts[category] * lengths[category])):])
    print('Total used samples: ', len(y_train)+len(y_test))
    print('Total training samples: ', len(y_train))
    print('Total testing samples: ', len(y_test))
    print('>> DONE Preparing Dataset')

    a = np.arange(20)
    print(a[:8])
    print(a[-8:])
    print(a[-0:])

    # shuffle again
    shuffler_train = np.random.permutation(len(X_train))
    shuffler_test = np.random.permutation(len(X_test))
    X_train, y_train, filenames_train = np.array(X_train)[shuffler_train], np.array(y_train)[shuffler_train], np.array(filenames_train)[shuffler_train]
    X_test, y_test, filenames_test = np.array(X_test)[shuffler_test], np.array(y_test)[shuffler_test], np.array(filenames_test)[shuffler_test]

    return X_train, X_test, y_train, y_test, filenames_train, filenames_test
