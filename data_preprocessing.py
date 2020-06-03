import cupsnbottles.feature_extraction as feature_extraction

import numpy as np
import os
import pickle as pkl
from PIL import Image
import h5py

path = 'dataset01/'


files = feature_extraction.get_files_of_type(path, type='g')
imgs = feature_extraction.read_images(files, os.path.join(path, 'images'))
feature_extraction.create_arbitrary_image_ds(os.path.join(path, 'images'), path)
