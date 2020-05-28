import cupsnbottles.load_cupsnbottles as load_cupsnbottles
import numpy as np
from sklearn import manifold

def load_gt_data(num_samples:int, path="cupsnbottles/" ):
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


def t_sne(X, dims=2, perplexity=30, learning_rate=200.0, n_iter=1000):
    """
    Calls t-SNE dimension reduction with default parameters. Can be adjusted.
    """
    tsne = manifold.TSNE(n_components=dims, init='random', perplexity=perplexity,
                         learning_rate = learning_rate,
                         n_iter=n_iter, n_iter_without_progress=300, method='barnes_hut',
                         random_state=0)
    return tsne.fit_transform(X)
