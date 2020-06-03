import cupsnbottles.load_cupsnbottles as load_cupsnbottles
import numpy as np
from sklearn import manifold
import os
import pickle as pkl
import pandas as pd

def load_gt_data(num_samples:int, path="cupsnbottles/" ):

    if path is not 'cupsnbottles/':
        X = load_features(path)
        print(X)
        label_names = ['Can01', 'Can02', 'Can03', 'Coke001', 'Coke002Cup', 'WaterBottle']

        df = []
        y = []
        y_encoded = []
        count = 0
        for x in sorted(os.listdir(os.path.join(path, 'images'))):
            if x.split(".")[-1] == "png":
                df.append(
                    {
                        'index': count,
                        'label': x.replace('.png', '')
                    }
                )
                count = count + 1
                for i, label in enumerate(label_names):
                    if label in name:
                        y.append(label)
                        y_encoded.append(i)
        pd.DataFrame(df)


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
        X = X[:num_samples]
        y = y[:num_samples]
        y_encoded = y_encoded[:num_samples]

    return X, y_encoded, y, label_names, df


def load_features(path):
    """load VGG19 feautures of object images from pkl file"""
    with open(os.path.join(path,'features.pkl'), 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        df = u.load()
    #index = np.array(df[0])
    #feats = np.array(df[1])
    print(df)
    f.close()

    return feats

def t_sne(X, dims=2, perplexity=30, learning_rate=200.0, n_iter=1000):
    """
    Calls t-SNE dimension reduction with default parameters. Can be adjusted.
    """
    tsne = manifold.TSNE(n_components=dims, init='random', perplexity=perplexity,
                         learning_rate = learning_rate,
                         n_iter=n_iter, n_iter_without_progress=300, method='barnes_hut',
                         random_state=0)
    return tsne.fit_transform(X)
