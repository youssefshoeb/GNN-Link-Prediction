import os
import urllib.request
import tarfile

"""
## Download Dataset
"""

urls = {'train': "https://bnn.upc.edu/download/ch21-training-dataset",
        'val': "https://bnn.upc.edu/download/ch21-validation-dataset",
        'test': "https://bnn.upc.edu/download/ch21-test-dataset-with-labels"}

os.makedirs('./data', exist_ok=True)
os.makedirs('./data/raw', exist_ok=True)
os.makedirs('./data/processed', exist_ok=True)

print("Downloading dataset...")
for k, v in urls.items():
    urllib.request.urlretrieve(v, f'./data/raw/{k}.tar.gz')


"""
## Extract Tar-Files
"""

print("Extracting Tar-Files...")
for k, v in urls.items():
    tar = tarfile.open(f'./data/raw/{k}.tar.gz')
    tar.extractall('./data/raw')
    tar.close()
