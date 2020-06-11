import os
import numpy as np
import pickle as pkl
from PIL import Image
import h5py


def load_properties(path):
    """Loads other properties of samples (i.e. labels) from pickle dump (also available via .csv file for other programming languages)"""
    with open(os.path.join(path,'properties.pkl'), 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        df = u.load()
    return df

def load_features(path):
    """load VGG19 feautures of object images from HDF file"""
    feat_path = os.path.join(path,'features.hdf')
    f = h5py.File(feat_path, 'r')
    index = np.array(f['index'])
    feats = np.array(f['feats'])
    f.close()

    return feats

def load_images(path,indices):
    """load images from files, indices are a list of indices like defined in properties file (df.index)"""
    imgs = []
    for i in indices:
        imgs.append(Image.open(os.path.join(path, 'images', i + '.jpg')))
    return imgs