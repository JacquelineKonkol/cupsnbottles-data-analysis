import cupsnbottles.feature_extraction as feature_extraction
import tools.basics as tools
import numpy as np
import argparse
import os
import PIL
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('path', help="Provide path to the dataset folder, in which images/ should be preprocessed.")
args = parser.parse_args()

# def clean_dataset():
#
#     # properties = tools.open_pkl(os.path.join(args.path, 'properties.pkl'))
#     # TODO in future: wenn die properties.csv file final ist und von Anfang an
#     # benutzt wird, dann müssen entsprechende Zeilen beim Löschen gedroppt werden
#
#     os.mkdir(os.path.join(args.path, 'deleted_images'))
#
#     dir = os.path.join(args.path, 'images')
#     for imagefile in sorted(os.listdir(dir)):
#         if imagefile.split(".")[-1] == "png":
#             img = PIL.Image.open(os.path.join(dir, imagefile))
#             # if image all black or all white
#             if not img.getbbox() or not PIL.ImageChops.invert(img).getbbox():
#                 os.rename(os.path.join(args.path, os.path.join('images', imagefile)),
#                           os.path.join(args.path, os.path.join('deleted_images', imagefile)))
#                 print(str(imagefile) + ' deleted.')
#
#     print('>> DONE cleaning dataset')

# def create_properties_csv(properties_df):
#     with open(os.path.join(args.path,'properties.pkl'),'wb') as f:
#         pkl.dump(properties_df,f)


#clean_dataset()
feature_extraction.create_arbitrary_image_ds(os.path.join(args.path, 'images'), args.path)
