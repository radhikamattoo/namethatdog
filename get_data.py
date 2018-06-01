# -*- coding: utf-8 -*-
# Radhika Mattoo
# Downloads and prepares the Stanford Dogs Dataset
from bs4 import BeautifulSoup
import scipy.io as sio
import requests
import tarfile
import os
import sys

# Downloads and untars necessary data
def get_data():
    # URLs for downloading data
    urls = [
        'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar',
        'http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar',
        'http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar'
    ]

    # Relative path of file destinations
    download_destinations = ['data/tars/images.tar', 'data/tars/lists.tar', 'data/annotations/annotation.tar']
    final_destinations = ['data/', 'data/lists', 'data/']

    # Download and untar files only if dirs don't exist
    for idx, url in enumerate(urls):
        # Create directories if they don't exist
        download_destination = download_destinations[idx]
        final_destination = final_destinations[idx]
        mkdirs([ download_destination, final_destination ])

        # Download data
        print 'Getting data from', url
        r = requests.get(url)
        with open(download_destination, 'wb') as f:
            f.write(r.content)

        # Untar
        print 'Untarring...'
        tar = tarfile.open(download_destination)
        tar.extractall(final_destination)
        tar.close()

    # Rename Images to images (from images.tar extraction)
    if os.path.exists('data/Images'):
        os.rename('data/Images', 'data/images')
    if os.path.exists('data/Annotation'):
        os.rename('data/Annotation', 'data/annotations')

# Splits images into train/test directories
def split_data():
    print '\nSplitting data into train/test directories'

    # Read in train/test lists
    train_dict = sio.loadmat('data/lists/train_list.mat')['annotation_list']
    test_dict = sio.loadmat('data/lists/test_list.mat')['annotation_list']
    train_list, test_list = convert_to_list(train_dict), convert_to_list(test_dict)
    mkdirs( ['data/train', 'data/val'] )

    # TODO: Read in annotations
    # Walk through annotations/ directory
    # Get dog breed from directory name
    # Make train/test dir with breed name
    # Walk through xml files in each directory
    # Check if file is in train or test and rename

    # for download_path in train_list:
    #     if os.path.exists(download_path):
    #         filename = os.path.basename(download_path)
    #         train_path = os.path.abspath(os.path.join('data/train/', filename))
    #         print train_path
    #         print download_path
    #         os.rename(download_path, train_path)
    #     else:
    #         print 'File not found: ', download_path
    # for download_path in test_list:
    #     if os.path.exists(download_path):
    #         filename = os.path.basename(download_path)
    #         test_path = os.path.abspath(os.path.join('data/test/', filename))
    #         os.rename(download_path, test_path)
    #     else:
    #         print 'File not found: ', download_path

# Create the given directories
def mkdirs(names):
    for name in names:
        dirname = os.path.dirname(name) if os.path.isfile(name) else name
        if not os.path.exists(dirname):
            os.makedirs(dirname)

# Collates items from the Matlab dict into a list
def convert_to_list(matlab_dict):
    file_list = []
    for item in matlab_dict:
        filepath = item[0][0]
        file_list.append( os.path.abspath('data/images/' + filepath + '.jpg') )
    return file_list


if __name__ == '__main__':
    if not os.path.exists('data/tars'):
        get_data()
    # if not os.path.exists('data/train') and not os.path.exists('data/test'):
    split_data()
