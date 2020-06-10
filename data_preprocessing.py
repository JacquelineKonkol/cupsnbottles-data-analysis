import cupsnbottles.feature_extraction as feature_extraction
import argparse

args = argparse.ArgumentParser()
args.add_argument(path, help="Provide path to the dataset folder, in which images/ should be preprocessed.")
args.parse_args()

feature_extraction.create_arbitrary_image_ds(os.path.join(path, 'images'), args.path)
