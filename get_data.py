# Radhika Mattoo, rm3485@nyu.edu
# This file downloads and prepares the Stanford Dogs Dataset
import requests
import scipy.io as sio
import tarfile
import os

# URLs for downloading data
urls = [
    'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar',
    'http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar'
]
# Relative path of file destinations
download_destinations = ['data/tars/images.tar', 'data/tars/lists.tar']
final_destinations = ['data/', 'data/lists']

# Downloads and untars necessary data
def get_data():
    # Download and untar files
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

# Helper function for creating directories
def mkdirs(dirnames):
    for dirname in dirnames:
        if not os.path.exists(os.path.dirname(dirname)):
            try:
                os.makedirs(os.path.dirname(dirname))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

if __name__ == '__main__':
    get_data()
