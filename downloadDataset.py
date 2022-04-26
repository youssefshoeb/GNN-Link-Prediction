import os
import urllib.request
import tarfile

urls = {'train': "https://bnn.upc.edu/download/ch21-training-dataset",
        'val': "https://bnn.upc.edu/download/ch21-validation-dataset",
        'test': "https://bnn.upc.edu/download/ch21-test-dataset-with-labels"
        }

"""
Download & extract dataset
"""


def download_dataset():
    print("Downloading Dataset...")
    os.makedirs('./dataset', exist_ok=True)
    for k, v in urls.items():
        urllib.request.urlretrieve(v, f'./dataset/{k}.tar.gz')


def extract_tarfiles():
    print("Extracting Files...")
    for k, _ in urls.items():
        tar = tarfile.open(f'./dataset/{k}.tar.gz')
        tar.extractall('./dataset')
        tar.close()


if __name__ == "__main__":
    # Download & extract dataset
    download_dataset()
    extract_tarfiles()
