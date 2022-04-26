import os
import torch
import torch_geometric
import datanetAPI
import re
import math
import csv

import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from models import QTBaseline


RAW_DIRS = {'train': './dataset/gnnet-ch21-dataset-train',
            'validation': './dataset/gnnet-ch21-dataset-validation',
            'test': './dataset/gnnet-ch21-dataset-test-with-labels'
            }

CONVERTED_DIRS = {'train': './dataset/converted_train',
                  'validation': './dataset/converted_validation',
                  'test': './dataset/converted_test'
                  }

BATCH_SIZE = {'train':8,'val':1}

class GNN21Dataset(torch_geometric.data.Dataset):
    """
    Convert pytorch Data objects to GNN input
    """

    def normalize(self, data):
        data["link"].x[:, 0] = (
            data["link"].x[:, 0] - 0.3546671) / 0.2083346
        data["link"].x[:, 1] = (
            data["link"].x[:, 1] - 0.16771736017268535) / 0.1974350417861857
        data["link"].x[:, 2] = (
            data["link"].x[:, 2] - 0.09862498490722958) / 0.179935315102362
        data["link"].x[:, 3] = (
            data["link"].x[:, 3] - 0.05104) / 0.06313
        data["link"].x[:, 4] = (
            data["link"].x[:, 4] - 0.35411) / 0.2075
        data["link"].x[:, 5] = (
            data["link"].x[:, 5] - 0.00066) / 0.00816
        data["path"].x[:, 0] = (
            data["path"].x[:, 0] - 0.6577772) / 0.4192159
        data["path"].x[:, 1] = (
            data["path"].x[:, 1] - 0.6578069) / 0.4192953
        data["path"].x[:, 2] = (
            data["path"].x[:, 2] - 0.6578076) / 0.4193256
        data["path"].x[:, 3] = (
            data["path"].x[:, 3] - 0.20152) / 0.18457

        # data["path"].y = (
        #    data["path"].y - 0) / (9.15503 - 0)

        return data

    def preprocess(self, data, converted_path):
        # Remove features that have same value across different nodes/simulations
        del data.p_SizeDist, data.p_TimeDist, data.p_ToS, data.p_time_ExpMaxFactor, data.p_TotalPktsGen, data.EqLambda, data.PktSize2, data.PktSize1, data.AvgPktSize
        del data.n_queueSizes, data.n_levelsQoS, data.n_schedulingPolicy

        # Path Attributes
        timeparams = [f'p_time_{a}' for a in ['AvgPktsLambda']]

        p_params = timeparams + ['p_PktsGen', 'p_AvgBw']

        data.p_AvgBw /= 1000.0

        data.P = torch.cat([getattr(data, a).view(-1, 1) for a in p_params], axis=1)

        mean_pkts_rate = data.p_time_AvgPktsLambda.mean().item()

        #  Global Attributes
        global_attrs = ['g_delay', 'g_packets', 'g_losses', 'g_AvgPktsLambda']

        for a in global_attrs:
            delattr(data, a)

        # Link attributes
        data.L = data.l_capacity.clone().view(-1, 1)

        # Get baseline features
        b_out, b_occup = self.baseline(data)

        # Create data object
        torch_data = torch_geometric.data.HeteroData()
        l_params = ['l_link_load', 'l_link_load2', 'l_link_load3']
        torch_data['link'].x = torch.cat([getattr(data, a).view(-1, 1) for a in l_params], axis=1)
        torch_data['link'].x = torch.cat([torch_data['link'].x,data.L/(mean_pkts_rate*10000)], axis=1) # link_capacity*


        p_params = ['p_time_AvgPktsLambda', 'p_PktsGen', 'p_AvgBw']
        torch_data['path'].x = torch.cat([getattr(data, a).view(-1, 1) for a in p_params], axis=1)
        torch_data['path'].x = torch.cat([torch_data['path'].x, torch_data['path'].x[:,0].view(-1,1) /mean_pkts_rate], axis=1) # p_time_AvgPktsLambda*
        torch_data['path'].x = torch.cat([torch_data['path'].x, torch_data['path'].x[:,1].view(-1,1) /mean_pkts_rate], axis=1)  # p_PktsGen*
        torch_data['path'].x = torch.cat([torch_data['path'].x, torch_data['path'].x[:,2].view(-1,1) /mean_pkts_rate], axis=1)  # p_AvgBw*

        num_node_nodes = (data.num_nodes - int(torch_data['path'].x.shape[0]) - int(torch_data['link'].x.shape[0]))
        torch_data['node'].x = torch.ones((num_node_nodes, 3))

        
        torch_data['link'].x = torch.cat([torch_data['link'].x, b_occup], axis=1)  # add baseline link features
        torch_data['path'].x = torch.cat([torch_data['path'].x, b_out.reshape((-1, 1))], axis=1)  # add baseline path feature

        # Label
        torch_data['path'].y = data['out_delay']

        # Get adjacency info
        torch_data['path', 'uses', 'link'].edge_index = data['p-l']
        torch_data['link', 'includes', 'path'].edge_index = data['l-p']
        torch_data['link', 'connects', 'node'].edge_index = data['l-n']
        torch_data['node', 'has', 'link'].edge_index = data['n-l']
        torch_data['path', 'is_connected', 'node'].edge_index = data['p-n']
        torch_data['node', 'is_used', 'path'].edge_index = data['n-p']

        # torch_data = self.normalize(torch_data)
        if converted_path is not None:
            torch.save(torch_data, converted_path)

        return torch_data

    def __init__(self, root_dir, convert_files, normalize=None, filenames=None):
        self.root_dir = root_dir
        self.convert_files = convert_files

        self.baseline = QTBaseline()

        if filenames is None:
            onlyfiles = [f for f in os.listdir(self.root_dir) if osp.isfile(osp.join(self.root_dir, f))]
            self.filenames = [f for f in onlyfiles if f.endswith('.pt')]
        else:
            self.filenames = filenames

    def __len__(self):
        """
        if self.root_dir == CONVERTED_DIRS['train']:
            return 1000 # len(self.filenames)
        else:
            return len(self.filenames)
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.filenames[idx]
        pt_path = osp.join(self.root_dir, filename)

        converted_dir = self.root_dir + '_processed'
        os.makedirs(converted_dir, exist_ok=True)

        converted_path = osp.join(converted_dir, filename)

        # if convert_files is False then load saved processed file to save time
        if self.convert_files:
            sample = torch.load(pt_path, map_location='cpu')
            sample = self.preprocess(sample, converted_path=converted_path)
        else:
            sample = torch.load(converted_path, map_location='cpu')

        if self.normalize:
            sample = self.normalize(sample)
        return sample


def preprocess_dataset():
    """
    Pass through the dataset to preprocess the dataset and save the processed objects
    """
    train_dataset = GNN21Dataset(root_dir=CONVERTED_DIRS['train'], convert_files=True)
    val_dataset = GNN21Dataset(root_dir=CONVERTED_DIRS['validation'], convert_files=True)
    test_dataset = GNN21Dataset(root_dir=CONVERTED_DIRS['test'], convert_files=True)

    train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=BATCH_SIZE['train'], shuffle=False)
    val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=BATCH_SIZE['val'], shuffle=False)
    test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=BATCH_SIZE['val'], shuffle=False)
    i = 0

    for _ in tqdm(train_loader):
        pass

    for _ in tqdm(val_loader):
        pass

    for _ in tqdm(test_loader):
        pass


def match_file(f):
    g = re.match("(validation|train|test)\_(\d+)\_(\d+).*", f).groups()
    g = [g[0], int(g[1]), int(g[2])]
    return g


def seperate_validation_dataset(filenames, path_to_original_dataset):

    matches = [match_file(f) for f in filenames]
    reader = datanetAPI.DatanetAPI(path_to_original_dataset)
    files_num = np.array([m[1] for m in matches], dtype=np.int32)
    samples_num = np.array([m[2] for m in matches], dtype=np.int32)

    all_paths = np.array(reader.get_available_files())

    df = pd.DataFrame(index=filenames, columns=['full_path', 'num_nodes', 'validation_setting'])
    df['full_path'] = all_paths[files_num, 0]
    df['sample_num'] = samples_num
    df['file_num'] = files_num

    df['num_nodes'] = np.array([osp.split(f)[-1] for f in df['full_path'].values], dtype=np.int32)

    if matches[0][0] in ['validation', 'test']:
        df['validation_setting'] = np.array([osp.split(f)[-2][-1] for f in df['full_path'].values], dtype=np.int32)
    else:
        df['validation_setting'] = -1

    df = df.sort_values(by=['validation_setting', 'num_nodes', 'file_num', 'sample_num'])
    return df


def initDataset(config):
    ds_val = GNN21Dataset(root_dir=CONVERTED_DIRS['validation'], convert_files=False)

    df_val = seperate_validation_dataset(ds_val.filenames, RAW_DIRS['validation'])

    df_val['filenames'] = df_val.index.values

    datasets = {"train": GNN21Dataset(root_dir=CONVERTED_DIRS['train'], convert_files=False, normalize=config["NORMALIZE_DATASET"]),
                "val": GNN21Dataset(root_dir=CONVERTED_DIRS['validation'], convert_files=False, normalize=config["NORMALIZE_DATASET"]),
                "test": GNN21Dataset(root_dir=CONVERTED_DIRS['test'], convert_files=False, normalize=config["NORMALIZE_DATASET"])}
    for i in range(3):
        which_files = list(df_val[df_val['validation_setting'] == i + 1]['filenames'].values)
        ds = GNN21Dataset(root_dir=CONVERTED_DIRS['validation'], convert_files=False, filenames=which_files)
        datasets[f'val_{i+1}'] = ds

    dataloaders = {}
    for k in datasets.keys():
        if k.startswith('train'):
            dataloaders[k] = torch_geometric.loader.DataLoader(datasets[k], batch_size=BATCH_SIZE['train'], shuffle=True)
        else:
            dataloaders[k] = torch_geometric.loader.DataLoader(datasets[k], batch_size=BATCH_SIZE['val'], shuffle=False)

    return datasets, dataloaders


def get_statistics():
    datasets, _ = initDataset()
    for k in datasets.keys():

        labels = pd.Series([], dtype="float64")
        link_load = pd.Series([], dtype="float64")
        link_load_2 = pd.Series([], dtype="float64")
        link_load_3 = pd.Series([], dtype="float64")
        link_capacity = pd.Series([], dtype="float64")
        link_baseline_1 = pd.Series([], dtype="float64")
        link_baseline_2 = pd.Series([], dtype="float64")
        link_baseline_3 = pd.Series([], dtype="float64")

        AvgPktsLambda = pd.Series([], dtype="float64")
        PktsGen = pd.Series([], dtype="float64")
        AvgBw = pd.Series([], dtype="float64")
        _AvgPktsLambda = pd.Series([], dtype="float64")
        _PktsGen = pd.Series([], dtype="float64")
        _AvgBw = pd.Series([], dtype="float64")
        path_baseline = pd.Series([], dtype="float64")
        
        for sample in tqdm(datasets[k]):
            s1 = pd.Series(sample['path'].y)
            labels = labels.append(s1, ignore_index=True)
            
            s1 = pd.Series(sample['link'].x[:, 0])
            link_load = link_load.append(s1, ignore_index=True)

            s1 = pd.Series(sample['link'].x[:, 1])
            link_load_2 = link_load_2.append(s1, ignore_index=True)

            s1 = pd.Series(sample['link'].x[:, 2])
            link_load_3 = link_load_3.append(s1, ignore_index=True)
            """
            s1 = pd.Series(sample['link'].x[:, 3])
            link_capacity = link_capacity.append(s1, ignore_index=True)

            s1 = pd.Series(sample['link'].x[:, 4])
            link_baseline_1 = link_baseline_1.append(s1, ignore_index=True)

            s1 = pd.Series(sample['link'].x[:, 5])
            link_baseline_2 = link_baseline_2.append(s1, ignore_index=True)

            s1 = pd.Series(sample['link'].x[:, 6])
            link_baseline_3 = link_baseline_3.append(s1, ignore_index=True)

            s1 = pd.Series(sample['path'].x[:, 0])
            AvgPktsLambda = AvgPktsLambda.append(s1, ignore_index=True)

            s1 = pd.Series(sample['path'].x[:, 1])
            PktsGen = PktsGen.append(s1, ignore_index=True)

            s1 = pd.Series(sample['path'].x[:, 2])
            AvgBw = AvgBw.append(s1, ignore_index=True)

            s1 = pd.Series(sample['path'].x[:, 3])
            _AvgPktsLambda = _AvgPktsLambda.append(s1, ignore_index=True)

            s1 = pd.Series(sample['path'].x[:, 4])
            _PktsGen = _PktsGen.append(s1, ignore_index=True)

            s1 = pd.Series(sample['path'].x[:, 5])
            _AvgBw = _AvgBw.append(s1, ignore_index=True)

            s1 = pd.Series(sample['path'].x[:, 6])
            path_baseline = path_baseline.append(s1, ignore_index=True)
            """



        labels.to_csv(f'{k}_labels.csv', header=None)
        
        link_load.to_csv(f'{k}_link_load.csv', header=None)
        link_load_2.to_csv(f'{k}_link_load_2.csv', header=None)
        link_load_3.to_csv(f'{k}_link_load_3.csv', header=None)
        """
        link_capacity.to_csv(f'{k}_link_capacity.csv', header=None)
        link_baseline_1.to_csv(f'{k}_link_baseline_1.csv', header=None)
        link_baseline_2.to_csv(f'{k}_link_baseline_2.csv', header=None)
        link_baseline_3.to_csv(f'{k}_link_baseline_3.csv', header=None)
        AvgPktsLambda.to_csv(f'{k}_AvgPktsLambda.csv', header=None)
        PktsGen.to_csv(f'{k}_PktsGen.csv', header=None)
        AvgBw.to_csv(f'{k}_AvgBw.csv', header=None)
        _AvgPktsLambda.to_csv(f'{k}__AvgPktsLambda_.csv', header=None)
        _PktsGen.to_csv(f'{k}__PktsGen_.csv', header=None)
        _AvgBw.to_csv(f'{k}__AvgBw_.csv', header=None)
        path_baseline.to_csv(f'{k}_path_baseline.csv', header=None)
        """


class Welford(object):
    """ Welford's algorithm for computing a running mean, std, and standard error
    """

    def __init__(self, lst=None):
        self.k = 0
        self.M = 0
        self.S = 0
        self.max = 0
        self.min = np.inf

        self.__call__(lst)

    def update(self, x):
        if x is None or isinstance(x, np.str_):
            return
        self.k += 1
        newM = self.M + (x - self.M) * 1. / self.k
        newS = self.S + (x - self.M) * (x - newM)
        self.M, self.S = newM, newS

        if x > self.max:
            self.max = x
        if x < self.min:
            self.min = x

    def consume(self, lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)

    def __call__(self, x):
        if hasattr(x, "__iter__"):
            self.consume(x)
        else:
            self.update(x)

    def to_csv(self,filename):

        # open the file in the write mode
        with open(filename, 'w') as f:
            # create the csv writer
            writer = csv.writer(f)

            # write the header
            writer.writerow(['Max', 'Min', 'Mean', 'STD'])

            # write the data
            writer.writerow([self.maximum, self.minimum, self.mean, self.std])

        pass

    @property
    def mean(self):
        return self.M

    @property
    def standard_error(self):
        if self.k == 0:
            return 0
        return self.std / math.sqrt(self.k)

    @property
    def maximum(self):
        return self.max

    @property
    def minimum(self):
        return self.min

    @property
    def std(self):
        if self.k == 1:
            return 0
        return math.sqrt(self.S / (self.k - 1))

    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)



def get_statistics2():
    datasets, _ = initDataset()
    for k in datasets.keys():

        
        #labels = Welford()
        #link_load = Welford()
        #link_load_2 = Welford()
        #link_load_3 = Welford()
        """
        link_capacity = Welford()
        
        """
        link_baseline_1 = Welford()
        link_baseline_2 = Welford()
        link_baseline_3 = Welford()
        
        AvgPktsLambda = Welford()
        PktsGen = Welford()
        AvgBw = Welford()
        """
        _AvgPktsLambda = Welford()
        _PktsGen = Welford()
        _AvgBw = Welford()
        """
        path_baseline = Welford()
        for sample in tqdm(datasets[k]):
            #labels(sample['path'].y)
            
            #link_load(sample['link'].x[:, 0])

            #link_load_2(sample['link'].x[:, 1])

            #link_load_3(sample['link'].x[:, 2])
            """
            link_capacity(sample['link'].x[:, 3])
            
            """
            link_baseline_1(sample['link'].x[:, 4])

            link_baseline_2(sample['link'].x[:, 5])

            link_baseline_3(sample['link'].x[:, 6])

            AvgPktsLambda(sample['path'].x[:, 0])
            PktsGen(sample['path'].x[:, 1])
            AvgBw(sample['path'].x[:, 2])
            """
            

            _AvgPktsLambda(sample['path'].x[:, 3])

            _PktsGen(sample['path'].x[:, 4])

            _AvgBw(sample['path'].x[:, 5])

            
            """
            path_baseline(sample['path'].x[:, 6])



        #labels.to_csv(f'{k}_labels.csv')
        
        #link_load.to_csv(f'{k}_link_load.csv')
        #link_load_2.to_csv(f'{k}_link_load_2.csv')
        #link_load_3.to_csv(f'{k}_link_load_3.csv')
        """
        link_capacity.to_csv(f'{k}_link_capacity.csv')
        """
        link_baseline_1.to_csv(f'{k}_link_baseline_1.csv')
        link_baseline_2.to_csv(f'{k}_link_baseline_2.csv')
        link_baseline_3.to_csv(f'{k}_link_baseline_3.csv')
        AvgPktsLambda.to_csv(f'{k}_AvgPktsLambda.csv')
        PktsGen.to_csv(f'{k}_PktsGen.csv')
        AvgBw.to_csv(f'{k}_AvgBw.csv')
        """
        _AvgPktsLambda.to_csv(f'{k}__AvgPktsLambda_.csv')
        _PktsGen.to_csv(f'{k}__PktsGen_.csv')
        _AvgBw.to_csv(f'{k}__AvgBw_.csv')
        """
        path_baseline.to_csv(f'{k}_path_baseline.csv')


if __name__ == "__main__":
    # Preprocess the dataset
    # preprocess_dataset()

    # Get Statistics
    get_statistics()
