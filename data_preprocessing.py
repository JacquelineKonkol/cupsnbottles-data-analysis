import cupsnbottles.feature_extraction as feature_extraction
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('path', help="Provide path to the dataset folder, in which images/ should be preprocessed.")
args = parser.parse_args()

feature_extraction.create_arbitrary_image_ds(os.path.join(args.path, 'images'), args.path)
