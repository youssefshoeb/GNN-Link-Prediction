import os
import urllib.request
import tarfile

import torch
import torch_geometric
import re
import datanetAPI
import multiprocessing
import tqdm

import numpy as np
import pandas as pd
import networkx as nx
import os.path as osp

from torch_geometric.data import Dataset
from torch_geometric.nn.conv import GINConv
from torch_geometric.utils import to_dense_adj
from typing import Callable, Any, Union
from torch.functional import Tensor
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.nn.conv.hetero_conv import HeteroConv
from torch_sparse import SparseTensor, matmul
from torch_scatter import scatter
from tqdm import tqdm


urls = {'train': "https://bnn.upc.edu/download/ch21-training-dataset",
        'val': "https://bnn.upc.edu/download/ch21-validation-dataset",
        'test': "https://bnn.upc.edu/download/ch21-test-dataset-with-labels"
        }

PROJECT_NAME = "PARANA2.0"

CONVERTED_DIRS = {'train': './dataset/converted_train',
                  'validation': './dataset/converted_validation',
                  'test': './dataset/converted_test'
                  }

RAW_DIRS = {'train': './dataset/gnnet-ch21-dataset-train',
            'validation': './dataset/gnnet-ch21-dataset-validation',
            'test': './dataset/gnnet-ch21-dataset-test'
            }

BATCH_SIZE = {'train': 8, 'val': 1}


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


"""
Generate pytorch files
"""


def simulation_to_networkX(network_graph, routing_matrix, traffic_matrix, performance_matrix, port_stats):
    """
    Convert dataset object to networkX graph
    """
    G = nx.DiGraph(network_graph)
    R = routing_matrix
    T = traffic_matrix
    D = performance_matrix
    P = port_stats

    D_G = nx.DiGraph()

    # Create node in graph corresponding to node in network
    for src in range(G.number_of_nodes()):
        D_G.add_node('n_{}'.format(src), **dict((f'n_{k}', v) for k, v in G.nodes[src].items()))

    # Go through each pair of nodes
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:

                # Create node in graph corresponding to edge in network
                if G.has_edge(src, dst):
                    D_G.add_node('l_{}_{}'.format(src, dst), l_capacity=G.edges[src, dst]['bandwidth'])

                    # Connect each edge-node with its source and destination
                    D_G.add_edge('n_{}'.format(src), 'l_{}_{}'.format(src, dst), edge_type=2)
                    D_G.add_edge('l_{}_{}'.format(src, dst), 'n_{}'.format(dst), edge_type=2)

                # Create node in graph corresponding to path in network
                for f_id in range(len(T[src, dst]['Flows'])):
                    if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                        dct_flows = dict((f'p_{k}', v) for k, v in T[src, dst]['Flows'][f_id].items())
                        # Some features are dicts so we need to pop them and put each individual element
                        dct_flows.pop('p_SizeDistParams')
                        dct_flows.pop('p_TimeDistParams')
                        dct_flows_size = dict((f'p_size_{k}', v) for k, v in T[src, dst]['Flows'][f_id]['SizeDistParams'].items())
                        dct_flows_time = dict((f'p_time_{k}', v) for k, v in T[src, dst]['Flows'][f_id]['TimeDistParams'].items())
                        dct_flows.update(dct_flows_size)
                        dct_flows.update(dct_flows_time)
                        dct_flows['out_delay'] = D[src, dst]['Flows'][f_id]['AvgDelay']

                        D_G.add_node('p_{}_{}_{}'.format(src, dst, f_id), **dct_flows)

                        # Connect path-nodes to link-nodes and node-nodes used
                        for _, (h_1, h_2) in enumerate([R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]):
                            _p = 'p_{}_{}_{}'.format(src, dst, f_id)
                            _l = 'l_{}_{}'.format(h_1, h_2)
                            _n1 = 'n_{}'.format(h_1)
                            _n2 = 'n_{}'.format(h_2)
                            if _n1 not in D_G[_p]:
                                D_G.add_edge(_p, _n1, edge_type=1)
                                D_G.add_edge(_n1, _p, edge_type=1)
                            if _n2 not in D_G[_p]:
                                D_G.add_edge(_p, _n2, edge_type=1)
                                D_G.add_edge(_n2, _p, edge_type=1)
                            D_G.add_edge(_p, _l, edge_type=0)
                            D_G.add_edge(_l, _p, edge_type=0)

    # Add link_load feature to each link
    link_loads = {}
    for node in D_G.nodes():
        if node.split('_')[0] == 'l':
            load = 0
            for src, _ in D_G.in_edges(node):
                if src.split('_')[0] == 'p':
                    load += D_G.nodes(data=True)[src]["p_AvgBw"]

            load /= D_G.nodes(data=True)[node]["l_capacity"]

            link_loads[node] = {
                "l_link_load": load, "l_link_load2": load * load, "l_link_load3": load * load * load}

    nx.set_node_attributes(D_G, link_loads)

    return D_G


def from_networkx(G):
    """
    Convert networkX object to torch object
    """

    # data dict used to create pytorch objects
    data = {}

    # key dict used to create edge index
    key_dict = {}
    paths_i = 0
    links_i = 0
    nodes_i = 0

    for key, feat_dict in G.nodes(data=True):
        if key.split('_')[0] == 'p':
            key_dict[key] = paths_i
            paths_i += 1

        elif key.split('_')[0] == 'l':
            key_dict[key] = links_i
            links_i += 1

        elif key.split('_')[0] == 'n':
            key_dict[key] = nodes_i
            nodes_i += 1

        for key, value in feat_dict.items():
            L = data.get(str(key), None)
            if L is None:
                data[key] = [value]
            else:
                L.append(value)

    for _, _, feat_dict in G.edges(data=True):
        for key, value in feat_dict.items():
            L = data.get(str(key), None)
            if L is None:
                data[key] = [value]
            else:
                L.append(value)

    # Get edges index
    data["p-l"] = []
    data["l-p"] = []
    data["l-n"] = []
    data["n-l"] = []
    data["p-n"] = []
    data["n-p"] = []
    for edge in G.edges:
        if edge[0].split("_")[0] == 'p' and edge[1].split("_")[0] == 'l':
            data["p-l"].append([key_dict[edge[0]], key_dict[edge[1]]])

        elif edge[0].split("_")[0] == 'l' and edge[1].split("_")[0] == 'p':
            data["l-p"].append([key_dict[edge[0]], key_dict[edge[1]]])

        elif edge[0].split("_")[0] == 'l' and edge[1].split("_")[0] == 'n':
            data["l-n"].append([key_dict[edge[0]], key_dict[edge[1]]])

        elif edge[0].split("_")[0] == 'n' and edge[1].split("_")[0] == 'l':
            data["n-l"].append([key_dict[edge[0]], key_dict[edge[1]]])

        elif edge[0].split("_")[0] == 'n' and edge[1].split("_")[0] == 'p':
            data["n-p"].append([key_dict[edge[0]], key_dict[edge[1]]])

        elif edge[0].split("_")[0] == 'p' and edge[1].split("_")[0] == 'n':
            data["p-n"].append([key_dict[edge[0]], key_dict[edge[1]]])

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data["p-l"] = data["p-l"].t().contiguous()
    data["l-p"] = data["l-p"].t().contiguous()
    data["l-n"] = data["l-n"].t().contiguous()
    data["n-l"] = data["n-l"].t().contiguous()
    data["p-n"] = data["p-n"].t().contiguous()
    data["n-p"] = data["n-p"].t().contiguous()

    G = nx.convert_node_labels_to_integers(G)
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()
    data['edge_index'] = edge_index.view(2, -1)

    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data


def name_to_id(s):
    s = s[0]
    if s == 'p':
        return 0
    elif s == 'l':
        return 1
    elif s == 'n':
        return 2
    raise Exception("node does not begin with p,l or n ")


def process_file(file_num, mode):
    """
    This method processes each simulation in the dataset and converts it to pytorch objects
    """
    os.makedirs(f'./dataset/converted_{mode}', exist_ok=True)

    reader = datanetAPI.DatanetAPI(RAW_DIRS[mode], intensity_values=[], topology_sizes=[], shuffle=False)
    reader._selected_tuple_files = [reader._all_tuple_files[file_num]]

    for i, sample in tqdm(enumerate(iter(reader))):
        G_copy = sample.get_topology_object().copy()

        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        D = sample.get_performance_matrix()
        P = sample.get_port_stats()
        graph = simulation_to_networkX(network_graph=G_copy,
                                       routing_matrix=R,
                                       traffic_matrix=T,
                                       performance_matrix=D,
                                       port_stats=P)

        data = from_networkx(graph)
        data.edge_index = data.edge_index.int()

        data.type = torch.as_tensor(np.array([name_to_id(name) for name in graph.nodes]))

        torch.save(data, f'./dataset/converted_{mode}/{mode}_{file_num}_{i}.pt')


def process_in_parallel(mode, max_proc=8):
    reader = datanetAPI.DatanetAPI(RAW_DIRS[mode], intensity_values=[], topology_sizes=[], shuffle=False)
    n_files = len(reader._all_tuple_files)
    pool = multiprocessing.Pool(processes=max_proc)
    for i in range(n_files):
        pool.apply_async(process_file, args=(i, mode))
    pool.close()
    pool.join()


def generate_files():
    print("Creating pytorch files (training)...")
    process_in_parallel('train', 8)

    print("Creating pytorch files (validation)...")
    process_in_parallel('validation', 4)

    print("Creating pytorch files (test)...")
    process_in_parallel('test', 4)


"""
Dataset
"""


class GNN21Dataset(Dataset):
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
        torch_data['link'].x = torch.cat([torch_data['link'].x, b_occup], axis=1)  # add baseline link features

        p_params = ['p_time_AvgPktsLambda', 'p_PktsGen', 'p_AvgBw']
        torch_data['path'].x = torch.cat([getattr(data, a).view(-1, 1) for a in p_params], axis=1)
        torch_data['path'].x = torch.cat([torch_data['path'].x, b_out.reshape((-1, 1))], axis=1)  # add baseline path feature

        num_node_nodes = (data.num_nodes - int(torch_data['path'].x.shape[0]) - int(torch_data['link'].x.shape[0]))
        torch_data['node'].x = torch.ones((num_node_nodes, 3))

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

    def __init__(self, root_dir, convert_files, filenames=None):
        self.root_dir = root_dir
        self.convert_files = convert_files

        self.baseline = QTBaseline()

        if filenames is None:
            onlyfiles = [f for f in os.listdir(self.root_dir) if osp.isfile(osp.join(self.root_dir, f))]
            self.filenames = [f for f in onlyfiles if f.endswith('.pt')]
        else:
            self.filenames = filenames

    def __len__(self):
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
            # sample = self.normalize(sample)

        return sample


def preprocess_dataset():
    """
    Pass through the dataset to preprocess the dataset and save the processed objects
    """
    train_dataset = GNN21Dataset(root_dir=CONVERTED_DIRS['train'], convert_files=True)
    val_dataset = GNN21Dataset(root_dir=CONVERTED_DIRS['validation'], convert_files=True)
    test_dataset = GNN21Dataset(root_dir=CONVERTED_DIRS['test'], convert_files=True)

    train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size=16, shuffle=False)
    val_loader = torch_geometric.loader.DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=2, shuffle=False)
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


def initDataset():
    ds_val = GNN21Dataset(root_dir=CONVERTED_DIRS['validation'], convert_files=False)

    df_val = seperate_validation_dataset(ds_val.filenames, RAW_DIRS['validation'])

    df_val['filenames'] = df_val.index.values

    datasets = {  # "train": GNN21Dataset(root_dir=CONVERTED_DIRS['train'], convert_files=False),
                  "val": GNN21Dataset(root_dir=CONVERTED_DIRS['validation'], convert_files=False),
                  "test": GNN21Dataset(root_dir=CONVERTED_DIRS['test'], convert_files=False)}
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


def separate_edge_timesteps(edge_index, edge_type):
    all_edges = [[] for _ in range(3)]
    for et in [0, 1, 2]:
        et_edges = edge_index[:, edge_type == et]

        init_tensor = torch.cat([torch.ones(1, device=et_edges.device).long(), torch.diff(et_edges[0, :])], dim=0)
        init_tensor = torch.clip(torch.abs(init_tensor), 0., 1.)

        init_tensor = 1 - init_tensor

        count_tensor = torch.nonzero(1 - init_tensor).view(-1)
        init_tensor[count_tensor[1:]] = -torch.diff(count_tensor) + 1

        init_tensor = init_tensor.cumsum(axis=0)

        ensure_stable = torch.linspace(start=0.0, end=0.5, steps=init_tensor.shape[0], device=init_tensor.device)
        encountered_order = torch.sort(init_tensor + ensure_stable)[1]
        et_edges = et_edges[:, encountered_order]

        _, vals = torch.unique(init_tensor, return_counts=True)
        vs = [x for x in torch.split_with_sizes(et_edges, tuple(vals), dim=1)]

        all_edges[et] = vs

    return all_edges


"""
Model
"""


class QTBaseline(torch.nn.Module):
    def __init__(self, num_iterations=3, G_dim=4, P_dim=3, L_dim=1, **kwargs):
        super(QTBaseline, self).__init__(**kwargs)
        self.num_iterations = num_iterations
        self.G_dim = G_dim
        self.P_dim = P_dim
        self.L_dim = L_dim

        self.H = 2
        self.H_p = 2
        self.H_l = 2
        self.H_n = 2

    def forward(self, data):
        edge_index = data.edge_index.long()
        edge_type = data.edge_type.clone()

        is_p = data.type == 0
        is_l = data.type == 1

        all_edges = separate_edge_timesteps(edge_index, edge_type)

        # Each element $i$ of pl_by_time contains all edges that occur at position $i$ in some path.
        pl_at_time = all_edges[0]

        edges_pl = edge_index[:, edge_type == 0]

        G_dim, P_dim, L_dim = self.G_dim, self.P_dim, self.L_dim
        H_n, H_p, H_l = self.H_n, self.H_p, self.H_l

        P = data.P * torch.ones(data.type.shape, device='cpu')[is_p].view(-1, 1)
        L = data.L * torch.ones(data.type.shape, device='cpu')[is_l].view(-1, 1)
        L = L / 1000

        cnt = 0
        cnt, _ = cnt + H_n, slice(cnt, cnt + H_n)
        cnt, node_og = cnt + G_dim, slice(cnt, cnt + G_dim)
        cnt, _ = cnt + H_l, slice(cnt, cnt + H_l)
        cnt, link_og = cnt + L_dim, slice(cnt, cnt + L_dim)
        cnt, _ = cnt + H_p, slice(cnt, cnt + H_p)
        cnt, path_og = cnt + P_dim, slice(cnt, cnt + P_dim)

        cnt = 0
        cnt, _ = cnt + H_n + G_dim, slice(cnt, cnt + H_n + G_dim)
        cnt, _ = cnt + H_l + L_dim, slice(cnt, cnt + H_l + L_dim)
        cnt, _ = cnt + H_p + P_dim, slice(cnt, cnt + H_p + P_dim)

        X = torch.zeros(data.type.size(0), H_n + G_dim + H_p + P_dim + H_l + L_dim, device='cpu')
        X[:, node_og] = 1
        X[is_l, link_og] = L
        X[is_p, path_og] = P

        #  Get Average bandwidth
        A = X[:, path_og.stop - 2].view(-1,).clone()  # Avg pkts sent
        blocking_probs = 0.3 * torch.ones_like(A)

        max_numpaths = len(pl_at_time)
        T = torch.zeros(X.size(0), device=A.device)
        rhos = torch.zeros(X.size(0), device=A.device)

        #  \trafic[k]_{i}: traffic passing on some edge that appears in order k at path
        def update_traffic(T, A, pl_at_time, blocking_probs):
            T = torch.zeros_like(T)

            for k in range(max_numpaths):
                if k == 0:
                    traffic = A.clone()
                else:
                    prev_paths = pl_at_time[k - 1][0, :]
                    prev_edges = pl_at_time[k - 1][1, :]
                    prev_edges_block_probs = torch.gather(blocking_probs, dim=0, index=prev_edges)

                    traffic[prev_paths] *= (1.0 - prev_edges_block_probs)

                which_paths = pl_at_time[k][0, :]
                which_edges = pl_at_time[k][1, :]
                T += scatter(src=torch.gather(traffic, 0, which_paths),
                             index=which_edges, dim=0, dim_size=X.size(0), reduce='sum')

            return T

        B = buffer_size = 32

        def update_blocking_probs(T, blocking_probs):
            blocking_probs = 0.0 * blocking_probs
            rhos = 0.0 * blocking_probs
            rhos[is_l] = T[is_l] / X[is_l, link_og.start]

            blocking_probs_num = (1.0 - rhos) * torch.pow(rhos, buffer_size)
            blocking_probs_den = 1.0 - torch.pow(rhos, buffer_size + 1)
            return blocking_probs_num / (blocking_probs_den + 1e-08)

        for _ in range(self.num_iterations):
            T = update_traffic(T, A, pl_at_time, blocking_probs)

            blocking_probs = update_blocking_probs(T, blocking_probs)

            rhos = T[is_l] / (X[is_l, link_og.start])
            pi_0 = (1 - rhos) / (1 - torch.pow(rhos, B + 1))
            res = 1 * pi_0
            for j in range(32):
                pi_0 = pi_0 * rhos
                res += (j + 1) * pi_0

            res = res / 32

        L = res

        #  Predict delay using predicted average utilization for each link in the path
        X = torch.zeros(X.size(0), device=X.device)
        data_L = data.L.squeeze(-1)
        link_capacity = data_L * torch.ones(data.type.shape, device='cpu')[is_l]  # * data.mean_pkts_rate[is_l]
        X[is_l] = L.squeeze(-1) * 32000.0 / link_capacity
        E = torch.gather(X, index=edges_pl[1, :], dim=0)
        res = scatter(src=E, index=edges_pl[0, :], dim=0, dim_size=X.size(0), reduce='sum')
        res = res[is_p]
        return res, torch.cat([x.view(-1, 1) for x in [L, rhos, pi_0]], axis=1)


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


def compute_identity(edge_index, num_layers):
    adj = to_dense_adj(edge_index)[0]
    diag_all = [torch.diag(adj)]
    adj_power = adj
    for _ in range(1, num_layers):
        adj_power = adj_power @ adj
        diag_all.append(torch.diag(adj_power))
    return torch.stack(diag_all, dim=1)


class GINConv(torch_geometric.nn.conv.MessagePassing):
    """
    This is the implmentation of GINConv from PytorchGeometric version: 2.0.2 I only added the concatination feature to it
    """
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False, concat: bool = False,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GINConv, self).__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        self.concat = concat
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1]
        if x_r is not None:
            if self.concat:
                out = torch.cat((out, (1 + self.eps) * x_r), 1)
            else:
                out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class GINLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, concat: bool = False) -> None:
        super().__init__()

        # Mlp that maps node features x of shape [-1, in_channels] to shape [-1, out_channels] )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.PReLU()
        )

        self.conv = GINConv(self.mlp, eps=0, train_eps=True, concat=concat)

    def forward(self, x: Tensor, edge_index: Adj):

        return self.conv(x, edge_index)


class HetroGIN(torch.nn.Module):
    def __init__(self, input_channels: dict):
        super().__init__()
        embedding_size = 8
        self.num_layers = 1

        self.convs = torch.nn.ModuleList()

        # Message Passing
        # first conv layer
        self.convs.append(HeteroConv({
            ('path', 'uses', 'link'): GINLayer(input_channels['path'] + input_channels['link'], embedding_size, concat=True),
            ('link', 'includes', 'path'): GINLayer(input_channels['link'] + input_channels['path'], embedding_size, concat=True),
            ('link', 'connects', 'node'): GINLayer(input_channels['link'] + input_channels['node'], embedding_size, concat=True),
            ('node', 'has', 'link'): GINLayer(input_channels['node'] + input_channels['link'], embedding_size, concat=True)}, aggr='sum'))

        # remaining conv layers
        for _ in range(self.num_layers - 1):
            self.convs.append(HeteroConv({
                ('path', 'uses', 'link'): GINLayer(embedding_size, embedding_size),
                ('link', 'includes', 'path'): GINLayer(embedding_size, embedding_size),
                ('link', 'connects', 'node'): GINLayer(embedding_size, embedding_size),
                ('node', 'has', 'link'): GINLayer(embedding_size, embedding_size)}, aggr='sum'))

        # Readout Layer
        self.readout = torch.nn.Sequential(torch.nn.Linear(embedding_size + input_channels['path'], 128),
                                           torch.nn.PReLU(),
                                           torch.nn.Linear(128, 32),
                                           torch.nn.PReLU(),
                                           torch.nn.Linear(32, 1)
                                           )

    def forward(self, x_dict, edge_index_dict):

        origin_input = x_dict.copy()
        # message passing
        for i in range(self.num_layers):
            x_dict = self.convs[i](x_dict, edge_index_dict)

        # readout
        x = torch.cat((x_dict['path'], origin_input['path']), 1)

        x = self.readout[i](x)

        return x


"""
Train
"""


def save_best_model(model):
    os.makedirs("./runs", exist_ok=True)

    print("Saving new best model ...")
    model = model.to("cpu")

    if not os.path.exists(f'./runs/{PROJECT_NAME}'):
        os.mkdir(f'runs/{PROJECT_NAME}')
    torch.save(model.state_dict(), f'./runs/{PROJECT_NAME}/best_model.pth')
    model = model.cuda()


def MAPE(preds, actuals):
    return 100.0 * torch.mean(torch.abs((preds - actuals) / actuals))


def train_one_epoch(epoch, dataloader, model):
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0

    opt = torch.optim.Adam(lr=1e-3, params=model.parameters())

    # Enumerate over the data
    total = len(dataloader)
    for sample in tqdm(dataloader, total=total):
        # Train model
        with torch.set_grad_enabled(True):

            # Reset Gradients
            opt.zero_grad()

            # Get Model Output
            out = model(sample.x_dict, sample.edge_index_dict)

            # Calculate loss and gradients
            label = sample['path'].y.reshape(-1, 1)
            loss_value = MAPE(out, label)

            loss = torch.sqrt(loss_value)
            loss.backward()
            opt.step()

            # Update Tracking
            running_loss += loss_value.item()
            step += 1
            # all_preds.append(out.cpu().detach().numpy())
            # all_labels.append(label.cpu().detach().numpy())

    # Calculate MAPE
    # all_preds = np.concatenate(all_preds).ravel()
    # all_labels = np.concatenate(all_labels).ravel()

    # mape_value = MAPE(torch.from_numpy(all_preds), torch.from_numpy(all_labels))
    average_loss = running_loss / step

    print(f"Epoch {epoch+1} | Train Loss {average_loss}")

    # print(f"MAPE-Train: {mape_value}")

    torch.cuda.empty_cache()

    return


def test(epoch, dataloader, model, mode):
    all_preds = []
    all_labels = []

    running_loss = 0.0
    step = 0

    with torch.no_grad():
        # Enumerate over the data
        for sample in tqdm(dataloader):
            with torch.set_grad_enabled(False):
                out = model(sample.x_dict, sample.edge_index_dict)

                # Get label
                label = sample['path'].y.reshape(-1, 1)
                loss_value = MAPE(out, label)

                # Update Tracking
                running_loss += loss_value.item()
                step += 1
                # all_preds.append(out.cpu().detach().numpy())
                # all_labels.append(label.cpu().detach().numpy())

        # all_preds = np.concatenate(all_preds).ravel()
        # all_labels = np.concatenate(all_labels).ravel()

        # mape_value = MAPE(torch.from_numpy(all_preds), torch.from_numpy(all_labels))
        average_loss = running_loss / step

        print(f"Epoch {epoch+1} | {mode} Loss {average_loss}")

        # print(f"MAPE-{mode}: {mape_value}")

        return average_loss


def train():
    print("Loading Dataset...")
    datasets, dataloaders = initDataset()

    print("Loading model...")
    input_channels = {'link': datasets["train"][0]['link']['x'].shape[1], 'path': datasets["train"][0]['path']['x'].shape[1], 'node': datasets["train"][0]['node']['x'].shape[1]}
    model = HetroGIN(input_channels=input_channels)

    num_epochs = 10
    step = 0

    # Start training
    best_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        # Train
        train_one_epoch(epoch, dataloaders['train'], model)

        model.eval()
        # Evaluation on validation set 1
        test(epoch, dataloaders['val_1'], model, "Validation_1")

        # Evaluation on validation set 2
        test(epoch, dataloaders['val_2'], model, "Validation_2")

        # Evaluation on validation set 3
        test(epoch, dataloaders['val_3'], model, "Validation_3")

        # Evaluation on validation set
        loss = test(epoch, dataloaders['val'], model, "Validation")

        # Update best loss
        if loss < best_loss:
            best_loss = loss
            save_best_model(model)


def test_baseline():
    print(f"Torch version: {torch.__version__}")
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Torch geometric version: {torch_geometric.__version__}")

    _, dataloaders = initDataset()

    # Train
    all_preds = []
    all_labels = []

    for sample in tqdm(dataloaders['test'], total=len(dataloaders['test'])):
        with torch.set_grad_enabled(False):
            b_out = sample['path'].x[:, 3]

            all_preds.append(torch.squeeze(b_out).cpu().detach().numpy())
            all_labels.append(sample['path'].y.cpu().detach().numpy())

    all_preds = torch.from_numpy(np.concatenate(all_preds).ravel())
    all_labels = torch.from_numpy(np.concatenate(all_labels).ravel())

    loss = MAPE(all_preds, all_labels)
    print('Test', loss)

    torch.cuda.empty_cache()

    """
    Training tensor(10.5695)
    Val tensor(10.0665)
    Val_1 tensor(11.2141)
    Val_2 tensor(10.7487)
    Val_3 tensor(9.3497)
    Test tensor(9.3962)
    """


def evaluate(filename):
    datasets, dataloaders = initDataset()

    print("Loading model...")
    input_channels = {'link': datasets["train"][0]['link']['x'].shape[1], 'path': datasets["train"][0]['path']['x'].shape[1], 'node': datasets["train"][0]['node']['x'].shape[1]}
    model = HetroGIN(input_channels=input_channels)

    model.load_state_dict(torch.load(f'./runs/{filename}//best_model.pth'))

    model.eval()

    for _, sample in tqdm(enumerate(dataloaders['test']), total=len(dataloaders['test'])):
        with torch.set_grad_enabled(False):
            out = model(sample.x_dict, sample.edge_index_dict)

            # Get label
            label = sample['path'].y.reshape(-1, 1)
            loss_value = MAPE(out, label)

            # Update Tracking
            running_loss += loss_value.item()
            step += 1

    average_loss = running_loss / step

    print('Test', average_loss)


if __name__ == "__main__":

    # Download & extract dataset
    download_dataset()
    extract_tarfiles()

    # Generate torch files
    generate_files()

    # Preprocess the dataset
    preprocess_dataset()

    # Test Baseline
    test_baseline()

    # Train
    train()

    # Test Final Model
    evaluate(PROJECT_NAME)
