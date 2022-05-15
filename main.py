import torch
import random
import numpy as np
import argparse
import json

from downloadDataset import download_dataset, extract_tarfiles
from generateFiles import generate_files
from dataset import preprocess_dataset,get_statistics2
from train import train

def set_random_seed(seed):
    # Ensure deterministic behavior by setting random seed
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False

    return

if __name__ == "__main__":
    # instantiate parser
    parser = argparse.ArgumentParser(description='Required files for training.')

    # Required hyperparameters path argument
    parser.add_argument('--config_file', type=str,
                        help='Config json file containing hyperparameters')

    args = parser.parse_args()
    # load hyperparameters
    with open(args.config_file) as data_file:
        json_config = json.load(data_file)

    # Download & extract dataset
    # download_dataset()
    # extract_tarfiles()

    # Generate torch files
    # generate_files()

    # Preprocess the dataset
    # preprocess_dataset()

    
    set_random_seed(json_config['SEED'])

    # Test Baseline
    # test_baseline(json_config)

    # Train
    train(json_config)  # Remember to change project name in config
    

