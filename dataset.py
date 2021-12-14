import torch
from torch.jit import Error
import torch_geometric

import numpy as np
import networkx as nx
import datanetAPI
import tqdm
import os

import pickle
import math


class Welford(object):
    """ Implements Welford's algorithm for computing a running mean
    and standard deviation as described at:
        http://www.johndcook.com/standard_deviation.html
    can take single values or iterables
    Properties:
        mean    - returns the mean
        std     - returns the std
        meanfull- returns the mean and std of the mean
    Usage:
        >>> foo = Welford()
        >>> foo(range(100))
        >>> foo
        <Welford: 49.5 +- 29.0114919759>
        >>> foo([1]*1000)
        >>> foo
        <Welford: 5.40909090909 +- 16.4437417146>
        >>> foo.mean
        5.409090909090906
        >>> foo.std
        16.44374171455467
        >>> foo.meanfull
        (5.409090909090906, 0.4957974674244838)
    """

    def __init__(self, lst=None):
        self.k = 0
        self.M = 0
        self.S = 0

        self.__call__(lst)

    def update(self, x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M) * 1. / self.k
        newS = self.S + (x - self.M) * (x - newM)
        self.M, self.S = newM, newS

    def consume(self, lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)

    def __call__(self, x):
        if hasattr(x, "__iter__"):
            self.consume(x)
        else:
            self.update(x)

    @property
    def mean(self):
        return self.M

    @property
    def meanfull(self):
        return self.mean, self.std / math.sqrt(self.k)

    @property
    def std(self):
        if self.k == 1:
            return 0
        return math.sqrt(self.S / (self.k - 1))

    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)


def generator(data_dir, intensity_values=[], topology_sizes=[], shuffle=False):
    """
    This function uses the DatanetAPI to output a single sample of the data.
    """
    tool = datanetAPI.DatanetAPI(
        data_dir, intensity_values, topology_sizes, shuffle=shuffle)
    it = iter(tool)
    num_samples = 0

    for sample in tqdm.tqdm(it):
        G_copy = sample.get_topology_object().copy()
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        D = sample.get_performance_matrix()
        P = sample.get_port_stats()
        HG = input_to_networkx(network_graph=G_copy,
                               routing_matrix=R,
                               traffic_matrix=T,
                               performance_matrix=D,
                               port_stats=P)
        num_samples += 1
        yield HG  # G_copy,R,P,D#hypergraph_to_input_data(HG)


def input_to_networkx(network_graph, routing_matrix, traffic_matrix, performance_matrix, port_stats):
    G = nx.DiGraph(network_graph)
    R = routing_matrix
    T = traffic_matrix
    D = performance_matrix
    P = port_stats
    # EDGE TYPES: 0 - path to link;  1- link to path; 2 - link to link;
    # All node features are added to the links in D_G_A

    # Create a directed Networkx object
    D_G_N = nx.DiGraph()  # directed graph with nodes
    D_G = nx.DiGraph()

    # Add all nodes to the graph
    for src in range(G.number_of_nodes()):
        D_G_N.add_node('n_{}'.format(src), **dict((f'n_{k}', v)
                       for k, v in G.nodes[src].items()))

    # Iterate over all nodes in the graph
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                if G.has_edge(src, dst):
                    # Add links to the graph as nodes
                    D_G.add_node('l_{}_{}'.format(src, dst),
                                 l_bandwidth=G.edges[src, dst]['bandwidth'])
                    src_stats = dict((f'n_{k}', v)
                                     for k, v in G.nodes[src].items())
                    link_stats = dict((f'l_{k}', v)
                                      for k, v in G.edges[src, dst].items())
                    link_stats.update(src_stats)

                    D_G_N.add_node('l_{}_{}'.format(src, dst), **link_stats)

                    D_G_N.add_edge('n_{}'.format(
                        src), 'l_{}_{}'.format(src, dst))
                    D_G_N.add_edge('l_{}_{}'.format(
                        src, dst), 'n_{}'.format(dst))

                # Iterate over all flows
                for f_id in range(len(T[src, dst]['Flows'])):

                    # Add paths to the graph as nodes
                    if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                        dct_flows = dict(
                            (f'p_{k}', v) for k, v in T[src, dst]['Flows'][f_id].items())

                        dct_flows_N = dict(
                            (f'p_{k}', v) for k, v in T[src, dst]['Flows'][f_id].items())

                        # Flatten dict of dicts to just a dict
                        dct_flows.pop('p_SizeDistParams')
                        dct_flows_N.pop('p_SizeDistParams')
                        dct_flows.pop('p_TimeDistParams')
                        dct_flows_N.pop('p_TimeDistParams')
                        dct_flows_size = dict(
                            (f'p_size_{k}', v) for k, v in T[src, dst]['Flows'][f_id]['SizeDistParams'].items())
                        dct_flows_time = dict(
                            (f'p_time_{k}', v) for k, v in T[src, dst]['Flows'][f_id]['TimeDistParams'].items())
                        dct_flows.update(dct_flows_size)
                        dct_flows_N.update(dct_flows_size)
                        dct_flows.update(dct_flows_time)
                        dct_flows_N.update(dct_flows_time)
                        dct_flows['p_AvgDelay'] = D[src,
                                                    dst]['Flows'][f_id]['AvgDelay']
                        dct_flows_N['p_AvgDelay'] = D[src,
                                                      dst]['Flows'][f_id]['AvgDelay']

                        # remove some features we already know have no effect
                        dct_flows.pop('p_ToS')
                        dct_flows.pop('p_time_ExpMaxFactor')
                        dct_flows.pop('p_TimeDist')
                        dct_flows.pop('p_SizeDist')
                        dct_flows.pop('p_size_AvgPktSize')
                        dct_flows.pop('p_size_PktSize1')
                        dct_flows.pop('p_size_PktSize2')
                        # maybe has an effect ?
                        dct_flows.pop('p_TotalPktsGen')

                        # divide remaing features by p_time_AvgPktsLambda
                        dct_flows['p_PktsGen'] = dct_flows['p_PktsGen'] / \
                            dct_flows['p_time_AvgPktsLambda']
                        dct_flows['p_AvgBw'] = dct_flows['p_AvgBw'] / \
                            dct_flows['p_time_AvgPktsLambda']
                        dct_flows['p_time_EqLambda'] = dct_flows['p_time_EqLambda'] / \
                            dct_flows['p_time_AvgPktsLambda']

                        temp_dict = {}
                        temp_dict['p_time_EqLambda*'] = dct_flows['p_time_EqLambda']
                        temp_dict['p_AvgBw*'] = dct_flows['p_AvgBw']
                        temp_dict['p_PktsGen*'] = dct_flows['p_PktsGen']
                        dct_flows_N.update(temp_dict)

                        print(dct_flows_N)

                        # raise features to square and cube
                        # dct_flows['p_PktsGen_2'] = dct_flows['p_PktsGen'] * dct_flows['p_PktsGen']
                        # dct_flows['p_AvgBw_2'] = dct_flows['p_AvgBw'] * dct_flows['p_AvgBw']
                        # dct_flows['p_time_EqLambda_2'] = dct_flows['p_time_EqLambda'] * dct_flows['p_time_EqLambda']
                        # dct_flows['p_PktsGen_3'] = dct_flows['p_PktsGen_2'] * dct_flows['p_PktsGen']
                        # dct_flows['p_AvgBw_3'] = dct_flows['p_AvgBw_2'] * dct_flows['p_AvgBw']
                        # dct_flows['p_time_EqLambda_3'] = dct_flows['p_time_EqLambda_2'] * dct_flows['p_time_EqLambda']

                        dct_flows.pop('p_time_AvgPktsLambda')

                        D_G.add_node('p_{}_{}_{}'.format(
                            src, dst, f_id), **dct_flows)

                        D_G_N.add_node('p_{}_{}_{}'.format(
                            src, dst, f_id), **dct_flows_N)

                        # Add edges between paths and links
                        for _, (h_1, h_2) in enumerate([R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]):
                            _p = 'p_{}_{}_{}'.format(src, dst, f_id)
                            _l = 'l_{}_{}'.format(h_1, h_2)
                            # _n1 =  'n_{}'.format(h_1)
                            # _n2 =  'n_{}'.format(h_2)
                            # if _n1 not in D_G[_p]:
                            #   D_G.add_edge(_p,_n1,edge_type=1)
                            # if _n2 not in D_G[_p]:
                            #   D_G.add_edge(_p,_n2,edge_type=1)
                            D_G.add_edge(_p, _l, edge_type=0)
                            D_G.add_edge(_l, _p, edge_type=1)

    # D_G.remove_nodes_from([node for node, out_degree in D_G.degree() if out_degree == 0])

    # Add edges between links connected by a node.
    for src, dst in D_G_N.edges():
        if src.split('_')[0] == 'l':
            for _, dst2 in D_G_N.out_edges(dst):
                edge = (src, dst2)
                if edge not in D_G.edges(src):
                    D_G.add_edge(src, dst2, edge_type=2)

    # Add link_load feature to each link
    link_loads = {}
    for node in D_G.nodes():
        if node.split('_')[0] == 'l':
            load = 0
            for src, _ in D_G.in_edges(node):
                if src.split('_')[0] == 'p':
                    load += D_G.nodes(data=True)[src]["p_AvgBw"]

            load /= D_G.nodes(data=True)[node]["l_bandwidth"]
            del D_G.nodes(data=True)[node]["l_bandwidth"]

            link_loads[node] = {
                "l_link_load": load, "l_link_load2": load * load, "l_link_load3": load * load * load}

    # for node in D_G.nodes():
    #    if node.split('_')[0] == 'p':
    #        del D_G.nodes(data=True)[node]["p_TotalPktsGen"]

    nx.set_node_attributes(D_G, link_loads)
    nx.set_node_attributes(D_G_N, link_loads)

    return D_G, D_G_N


def networkx_to_data(G):
    """
    Convert the networkx object to a feature matrix and edge index
    """
    feature_dict = {}
    key_dict = {}
    paths_i = 0
    links_i = 0

    # Get Path and Link features
    for _, (key, feat_dict) in enumerate(G.nodes(data=True)):
        if key.split('_')[0] == 'p':
            key_dict[key] = paths_i

            if paths_i == 0:
                feature_dict["paths"] = dict(
                    (k, np.array([])) for k, _ in feat_dict.items())

            path_features = dict((k, v) for k, v in feat_dict.items())

            for k, v in feature_dict["paths"].items():
                feature_dict["paths"][k] = np.append([v], [path_features[k]])

            paths_i += 1

        elif key.split('_')[0] == 'l':
            key_dict[key] = links_i

            if links_i == 0:
                feature_dict["links"] = dict(
                    (k, np.array([])) for k, _ in feat_dict.items())

            link_features = dict((k, v) for k, v in feat_dict.items())

            for k, v in feature_dict["links"].items():
                feature_dict["links"][k] = np.append([v], [link_features[k]])

            links_i += 1

    # Get edges index
    edge_dict = {"p-l": [], "l-p": [], "l-l": []}
    for edge in G.edges:
        if edge[0].split("_")[0] == 'p' and edge[1].split("_")[0] == 'l':
            edge_dict["p-l"].append([key_dict[edge[0]], key_dict[edge[1]]])

        elif edge[0].split("_")[0] == 'l' and edge[1].split("_")[0] == 'p':
            edge_dict["l-p"].append([key_dict[edge[0]], key_dict[edge[1]]])

        elif edge[0].split("_")[0] == 'l' and edge[1].split("_")[0] == 'l':
            edge_dict["l-l"].append([key_dict[edge[0]], key_dict[edge[1]]])

    edge_dict["p-l"] = torch.tensor(edge_dict["p-l"]).t().contiguous()
    edge_dict["l-p"] = torch.tensor(edge_dict["l-p"]).t().contiguous()
    edge_dict["l-l"] = torch.tensor(edge_dict["l-l"]).t().contiguous()

    return feature_dict, edge_dict


def transformation(x):
    """Apply a transformation over all the samples included in the dataset.
        Args:
            x (dict): predictor variable.
        Returns:
            x,y: The modified predictor/target variables.
        """
    x["links"]['l_link_load'] = (
        x["links"]['l_link_load'] - 0.0) / (1.9705873632629973 - 0.0)
    x["links"]['l_link_load2'] = (
        x["links"]['l_link_load2']) / (3.8832145562518123 - 0.0)
    x["links"]['l_link_load3'] = (
        x["links"]['l_link_load3']) / (7.652213533388749 - 0.0)
    x["paths"]['p_PktsGen'] = (
        x["paths"]['p_PktsGen'] - 0.7498462848223999) / (1.3038396633263194 - 0.7498462848223999)
    x["paths"]['p_AvgBw'] = (
        x["paths"]['p_AvgBw'] - 622.2114877920252) / (1346.7849033971643 - 622.2114877920252)
    x["paths"]['p_time_EqLambda'] = (
        x["paths"]['p_time_EqLambda'] - 999.9885313720984) / (1000.0249134258452 - 999.9885313720984)
    x["paths"]['p_AvgDelay'] = (
        x["paths"]['p_AvgDelay'] - 0.000418) / (9.15503 - 0.000418)

    # x['bandwith']=(x['bandwith']-bandwith_min)/(bandwith_max-bandwith_min)
    return x


class TorchDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename, intensity=[], topology_sizes=[], shuffle=False):
        'Initialization'
        self.generator_object = generator(
            filename, intensity, topology_sizes, shuffle=shuffle)

    def __iter__(self):
        'Generates one sample of data'
        return iter(self.generator_object)


class GNNC21Dataset(torch_geometric.data.Dataset):
    def __init__(self, root, filename, val=False, test=False, transform=None, pre_transform=None, stats=False, accum_stats=False, get_all_stats=False):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.val = val
        self.filename = filename
        self.accum_stats = accum_stats
        self.get_all_stats = get_all_stats

        if self.accum_stats:
            if not val and not test:
                raise IOError(
                    "accum_stats cannot be true while processing train dataset")
            if val:
                stats_file = "train"
            if test:
                stats_file = "train-val"

            with open(stats_file, "rb") as input_file:
                file = pickle.load(input_file)
                stats_array = file

        else:
            if stats:
                stats_mean_dict = {'l_link_load': Welford(), 'l_link_load2': Welford(), 'l_link_load3': Welford(), 'p_PktsGen*': Welford(), 'p_AvgBw*': Welford(), 'p_time_EqLambda*': Welford(),
                                   'n_levelsQoS': Welford(), 'n_schedulingPolicy': Welford(), 'n_queueSizes': Welford(), 'l_bandwidth': Welford(),
                                   'l_weight': Welford(), "l_port": Welford(), 'p_time_ExpMaxFactor': Welford(), 'p_time_AvgPktsLambda': Welford(),
                                   'p_time_EqLambda': Welford(), 'p_size_PktSize2': Welford(), 'p_size_PktSize1': Welford(), 'p_size_AvgPktSize': Welford(),
                                   'p_ToS': Welford(), 'p_TotalPktsGen': Welford(), "p_SizeDist": Welford(), 'p_TimeDist': Welford(),
                                   'p_AvgDelay': Welford(), 'p_PktsGen': Welford(), 'p_AvgBw': Welford(), 'p_time_EqLambda': Welford()}

                stats_max_dict = {'l_link_load': 0, 'l_link_load2': 0, 'l_link_load3': 0, 'p_PktsGen*': 0, 'p_AvgBw*': 0, 'p_time_EqLambda*': 0,
                                  'n_levelsQoS': 0, 'n_schedulingPolicy': 0, 'n_queueSizes': 0, 'l_bandwidth': 0,
                                  'l_weight': 0, "l_port": 0, 'p_time_ExpMaxFactor': 0, 'p_time_AvgPktsLambda': 0,
                                  'p_time_EqLambda': 0, 'p_size_PktSize2': 0, 'p_size_PktSize1': 0, 'p_size_AvgPktSize': 0,
                                  'p_ToS': 0, 'p_TotalPktsGen': 0, "p_SizeDist": 0, 'p_TimeDist': 0,
                                  'p_AvgDelay': 0, 'p_PktsGen': 0, 'p_AvgBw': 0, 'p_time_EqLambda': 0}

                stats_min_dict = {'l_link_load': np.inf, 'l_link_load2': np.inf, 'l_link_load3': np.inf, 'p_PktsGen*': np.inf, 'p_AvgBw*': np.inf, 'p_time_EqLambda*': np.inf,
                                  'n_levelsQoS': np.inf, 'n_schedulingPolicy': np.inf, 'n_queueSizes': np.inf, 'l_bandwidth': np.inf,
                                  'l_weight': np.inf, "l_port": np.inf, 'p_time_ExpMaxFactor': np.inf, 'p_time_AvgPktsLambda': np.inf,
                                  'p_time_EqLambda': np.inf, 'p_size_PktSize2': np.inf, 'p_size_PktSize1': np.inf, 'p_size_AvgPktSize': np.inf,
                                  'p_ToS': np.inf, 'p_TotalPktsGen': np.inf, "p_SizeDist": np.inf, 'p_TimeDist': np.inf,
                                  'p_AvgDelay': np.inf, 'p_PktsGen': np.inf, 'p_AvgBw': np.inf, 'p_time_EqLambda': np.inf}
                stats_array = [stats_mean_dict, stats_max_dict, stats_min_dict]
            else:
                stats_array = None

        self.stats = stats_array

        super(GNNC21Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        # i should just add the name of the dataset gnnet_data_set_training
        return ['dummy.csv']

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""

        if self.test:
            return [f'data_test_{i}.pt' for i in range(20)]
        elif self.val:
            return [f'data_val_{i}.pt' for i in range(20)]
        else:
            return [f'data_{i}.pt' for i in range(20)]

    def download(self):
        raise NotImplementedError("Download function not implemented")

    def process(self):
        dataset = TorchDataset(os.path.join(self.raw_dir, self.filename))
        index = 0

        for sample in dataset:
            feature_dict, edge_dict = networkx_to_data(sample[0])

            if self.stats:
                all_feature_dict, _ = networkx_to_data(sample[1])
                self._get_stats(all_feature_dict)

            feature_dict = transformation(feature_dict)

            # features, labels = sample
            # features, labels = transformation(features, labels)

            # Create data object
            data = torch_geometric.data.HeteroData()

            # Get node features and labels
            data['link'].x = self._get_link_features(
                feature_dict)  # [num_link, num_features_link]
            data['path'].x = self._get_path_features(
                feature_dict)  # [num_path, num_features_path]

            # Get adjacency info
            data['path', 'uses', 'link'].edge_index = edge_dict['p-l']
            data['link', 'includes', 'path'].edge_index = edge_dict['l-p']
            data['link', 'connects', 'link'].edge_index = edge_dict['l-l']

            # Adding reverse edges
            # data = torch_geometric.transforms.ToUndirected()(data)

            # Label
            # .reshape(-1,1) # [num_path] # I think the correct way is to reshape but even with no reshape I think it still works
            data['path'].y = self._get_label(feature_dict)

            if self.test:
                torch.save(data, os.path.join(
                    self.processed_dir, f'data_test_{index}.pt'))
            elif self.val:
                torch.save(data, os.path.join(
                    self.processed_dir, f'data_val_{index}.pt'))
            else:
                torch.save(data, os.path.join(
                    self.processed_dir, f'data_{index}.pt'))
            index += 1

        if self.stats:
            if self.test:
                if self.accum_stats:
                    with open("train-val-test", "wb") as output_file:
                        pickle.dump(self.stats, output_file)
                else:
                    with open("test", "wb") as output_file:
                        pickle.dump(self.stats, output_file)
            elif self.val:
                if self.accum_stats:
                    with open("train-val", "wb") as output_file:
                        pickle.dump(self.stats, output_file)
                else:
                    with open("val", "wb") as output_file:
                        pickle.dump(self.stats, output_file)
            else:
                with open("train", "wb") as output_file:
                    pickle.dump(self.stats, output_file)

    def _get_stats(self, features):
        for i in range(len(features['paths']['p_AvgDelay'])):
            self.stats[0]["p_AvgDelay"].update(
                features['paths']['p_AvgDelay'][i])
            self.stats[0]["p_PktsGen*"].update(
                features['paths']['p_PktsGen*'][i])
            self.stats[0]["p_AvgBw*"].update(features['paths']['p_AvgBw*'][i])
            self.stats[0]["p_time_EqLambda*"].update(
                features['paths']['p_time_EqLambda*'][i])
            self.stats[0]["p_time_ExpMaxFactor"].update(
                features['paths']['p_time_ExpMaxFactor'][i])
            self.stats[0]["p_time_AvgPktsLambda"].update(
                features['paths']['p_time_AvgPktsLambda'][i])
            self.stats[0]["p_size_PktSize2"].update(
                features['paths']['p_size_PktSize2'][i])
            self.stats[0]["p_size_PktSize1"].update(
                features['paths']['p_size_PktSize1'][i])
            self.stats[0]["p_size_AvgPktSize"].update(
                features['paths']['p_size_AvgPktSize'][i])
            self.stats[0]["p_ToS"].update(
                features['paths']['p_ToS'][i])
            self.stats[0]["p_TotalPktsGen"].update(
                features['paths']['p_TotalPktsGen'][i])
            self.stats[0]["p_SizeDist"].update(
                features['paths']['p_SizeDist'][i])
            self.stats[0]["p_TimeDist"].update(
                features['paths']['p_TimeDist'][i])
            self.stats[0]["p_PktsGen"].update(
                features['paths']['p_PktsGen'][i])
            self.stats[0]["p_AvgBw"].update(
                features['paths']['p_AvgBw'][i])
            self.stats[0]["p_time_EqLambda"].update(
                features['paths']['p_time_EqLambda'][i])

        if self.stats[1]["p_AvgDelay"] < max(features['paths']['p_AvgDelay']):
            self.stats[1]["p_AvgDelay"] = max(features['paths']['p_AvgDelay'])

        if self.stats[1]["p_PktsGen*"] < max(features['paths']['p_PktsGen*']):
            self.stats[1]["p_PktsGen*"] = max(features['paths']['p_PktsGen*'])

        if self.stats[1]["p_AvgBw*"] < max(features['paths']['p_AvgBw*']):
            self.stats[1]["p_AvgBw*"] = max(features['paths']['p_AvgBw*'])

        if self.stats[1]["p_time_EqLambda*"] < max(features['paths']['p_time_EqLambda*']):
            self.stats[1]["p_time_EqLambda*"] = max(
                features['paths']['p_time_EqLambda*'])

        if self.stats[1]["p_PktsGen"] < max(features['paths']['p_PktsGen']):
            self.stats[1]["p_PktsGen"] = max(features['paths']['p_PktsGen'])

        if self.stats[1]["p_AvgBw"] < max(features['paths']['p_AvgBw']):
            self.stats[1]["p_AvgBw"] = max(features['paths']['p_AvgBw'])

        if self.stats[1]["p_time_EqLambda"] < max(features['paths']['p_time_EqLambda']):
            self.stats[1]["p_time_EqLambda"] = max(
                features['paths']['p_time_EqLambda'])

        if self.stats[1]["p_TimeDist"] < max(features['paths']['p_TimeDist']):
            self.stats[1]["p_TimeDist"] = max(features['paths']['p_TimeDist'])

        if self.stats[1]["p_SizeDist"] < max(features['paths']['p_SizeDist']):
            self.stats[1]["p_SizeDist"] = max(features['paths']['p_SizeDist'])

        if self.stats[1]["p_TotalPktsGen"] < max(features['paths']['p_TotalPktsGen']):
            self.stats[1]["p_TotalPktsGen"] = max(
                features['paths']['p_TotalPktsGen'])

        if self.stats[1]["p_ToS"] < max(features['paths']['p_ToS']):
            self.stats[1]["p_ToS"] = max(features['paths']['p_ToS'])

        if self.stats[1]["p_size_AvgPktSize"] < max(features['paths']['p_size_AvgPktSize']):
            self.stats[1]["p_size_AvgPktSize"] = max(
                features['paths']['p_size_AvgPktSize'])

        if self.stats[1]["p_size_PktSize1"] < max(features['paths']['p_size_PktSize1']):
            self.stats[1]["p_size_PktSize1"] = max(
                features['paths']['p_size_PktSize1'])

        if self.stats[1]["p_size_PktSize2"] < max(features['paths']['p_size_PktSize2']):
            self.stats[1]["p_size_PktSize2"] = max(
                features['paths']['p_size_PktSize2'])

        if self.stats[1]["p_time_AvgPktsLambda"] < max(features['paths']['p_time_AvgPktsLambda']):
            self.stats[1]["p_time_AvgPktsLambda"] = max(
                features['paths']['p_time_AvgPktsLambda'])

        if self.stats[1]["p_time_ExpMaxFactor"] < max(features['paths']['p_time_ExpMaxFactor']):
            self.stats[1]["p_time_ExpMaxFactor"] = max(
                features['paths']['p_time_ExpMaxFactor'])

        if self.stats[2]["p_AvgDelay"] > min(features['paths']['p_AvgDelay']):
            self.stats[2]["p_AvgDelay"] = min(features['paths']['p_AvgDelay'])

        if self.stats[2]["p_PktsGen*"] > min(features['paths']['p_PktsGen*']):
            self.stats[2]["p_PktsGen*"] = min(features['paths']['p_PktsGen*'])

        if self.stats[2]["p_AvgBw*"] > min(features['paths']['p_AvgBw*']):
            self.stats[2]["p_AvgBw*"] = min(features['paths']['p_AvgBw*'])

        if self.stats[2]["p_time_EqLambda*"] > min(features['paths']['p_time_EqLambda*']):
            self.stats[2]["p_time_EqLambda*"] = min(
                features['paths']['p_time_EqLambda*'])

        if self.stats[2]["p_PktsGen"] > min(features['paths']['p_PktsGen']):
            self.stats[2]["p_PktsGen"] = min(features['paths']['p_PktsGen'])

        if self.stats[2]["p_AvgBw"] > min(features['paths']['p_AvgBw']):
            self.stats[2]["p_AvgBw"] = min(features['paths']['p_AvgBw'])

        if self.stats[2]["p_time_EqLambda"] > min(features['paths']['p_time_EqLambda']):
            self.stats[2]["p_time_EqLambda"] = min(
                features['paths']['p_time_EqLambda'])

        if self.stats[2]["p_TimeDist"] > min(features['paths']['p_TimeDist']):
            self.stats[2]["p_TimeDist"] = min(features['paths']['p_TimeDist'])

        if self.stats[2]["p_SizeDist"] > min(features['paths']['p_SizeDist']):
            self.stats[2]["p_SizeDist"] = min(features['paths']['p_SizeDist'])

        if self.stats[2]["p_TotalPktsGen"] > min(features['paths']['p_TotalPktsGen']):
            self.stats[2]["p_TotalPktsGen"] = min(
                features['paths']['p_TotalPktsGen'])

        if self.stats[2]["p_ToS"] > min(features['paths']['p_ToS']):
            self.stats[2]["p_ToS"] = min(features['paths']['p_ToS'])

        if self.stats[2]["p_size_AvgPktSize"] > min(features['paths']['p_size_AvgPktSize']):
            self.stats[2]["p_size_AvgPktSize"] = min(
                features['paths']['p_size_AvgPktSize'])

        if self.stats[2]["p_size_PktSize1"] > min(features['paths']['p_size_PktSize1']):
            self.stats[2]["p_size_PktSize1"] = min(
                features['paths']['p_size_PktSize1'])

        if self.stats[2]["p_size_PktSize2"] > min(features['paths']['p_size_PktSize2']):
            self.stats[2]["p_size_PktSize2"] = min(
                features['paths']['p_size_PktSize2'])

        if self.stats[2]["p_time_AvgPktsLambda"] > min(features['paths']['p_time_AvgPktsLambda']):
            self.stats[2]["p_time_AvgPktsLambda"] = min(
                features['paths']['p_time_AvgPktsLambda'])

        if self.stats[2]["p_time_ExpMaxFactor"] > min(features['paths']['p_time_ExpMaxFactor']):
            self.stats[2]["p_time_ExpMaxFactor"] = min(
                features['paths']['p_time_ExpMaxFactor'])

        for i in range(len(features['links']['l_link_load'])):
            self.stats[0]["l_link_load"].update(
                features["links"]['l_link_load'][i])
            self.stats[0]["l_link_load2"].update(
                features["links"]['l_link_load2'][i])
            self.stats[0]["l_link_load3"].update(
                features["links"]['l_link_load3'][i])
            self.stats[0]["n_levelsQoS"].update(
                features['links']['n_levelsQoS'][i])
            self.stats[0]["n_schedulingPolicy"].update(
                features['links']['n_schedulingPolicy'][i])
            self.stats[0]["n_queueSizes"].update(
                features['links']['n_queueSizes'][i])
            self.stats[0]["l_bandwidth"].update(
                features['links']['l_bandwidth'][i])
            self.stats[0]["l_weight"].update(
                features['links']['l_weight'][i])
            self.stats[0]["l_port"].update(
                features['links']['l_port'][i])

        if self.stats[1]["l_link_load"] < max(features["links"]['l_link_load']):
            self.stats[1]["l_link_load"] = max(
                features["links"]['l_link_load'])

        if self.stats[1]["l_link_load2"] < max(features["links"]['l_link_load2']):
            self.stats[1]["l_link_load2"] = max(
                features["links"]['l_link_load2'])

        if self.stats[1]["l_link_load3"] < max(features["links"]['l_link_load3']):
            self.stats[1]["l_link_load3"] = max(
                features["links"]['l_link_load3'])

        if self.stats[1]["n_levelsQoS"] < max(features["links"]['n_levelsQoS']):
            self.stats[1]["n_levelsQoS"] = max(
                features["links"]['n_levelsQoS'])

        if self.stats[1]["n_schedulingPolicy"] < max(features["links"]['n_schedulingPolicy']):
            self.stats[1]["n_schedulingPolicy"] = max(
                features["links"]['n_schedulingPolicy'])

        if self.stats[1]["n_queueSizes"] < max(features["links"]['n_queueSizes']):
            self.stats[1]["n_queueSizes"] = max(
                features["links"]['n_queueSizes'])

        if self.stats[1]["l_bandwidth"] < max(features["links"]['l_bandwidth']):
            self.stats[1]["l_bandwidth"] = max(
                features["links"]['l_bandwidth'])

        if self.stats[1]["l_weight"] < max(features["links"]['l_weight']):
            self.stats[1]["l_weight"] = max(
                features["links"]['l_weight'])

        if self.stats[1]["l_port"] < max(features["links"]['l_port']):
            self.stats[1]["l_port"] = max(
                features["links"]['l_port'])

        if self.stats[2]["l_link_load"] > min(features["links"]['l_link_load']):
            self.stats[2]["l_link_load"] = min(
                features["links"]['l_link_load'])

        if self.stats[2]["l_link_load2"] > min(features["links"]['l_link_load2']):
            self.stats[2]["l_link_load2"] = min(
                features["links"]['l_link_load2'])

        if self.stats[2]["l_link_load3"] > min(features["links"]['l_link_load3']):
            self.stats[2]["l_link_load3"] = min(
                features["links"]['l_link_load3'])

        if self.stats[2]["n_levelsQoS"] > min(features["links"]['n_levelsQoS']):
            self.stats[2]["n_levelsQoS"] = min(
                features["links"]['n_levelsQoS'])

        if self.stats[2]["n_schedulingPolicy"] > min(features["links"]['n_schedulingPolicy']):
            self.stats[2]["n_schedulingPolicy"] = min(
                features["links"]['n_schedulingPolicy'])

        if self.stats[2]["n_queueSizes"] > min(features["links"]['n_queueSizes']):
            self.stats[2]["n_queueSizes"] = min(
                features["links"]['n_queueSizes'])

        if self.stats[2]["l_bandwidth"] > min(features["links"]['l_bandwidth']):
            self.stats[2]["l_bandwidth"] = min(
                features["links"]['l_bandwidth'])

        if self.stats[2]["l_weight"] > min(features["links"]['l_weight']):
            self.stats[2]["l_weight"] = min(
                features["links"]['l_weight'])

        if self.stats[2]["l_port"] > min(features["links"]['l_port']):
            self.stats[2]["l_port"] = min(
                features["links"]['l_port'])

    def _get_link_features(self, features):
        """
        This will return a matrix / 2d array of the shape
        [Number of links, link Feature size]
        """
        n_links = len(features["links"][next(iter(features["links"]))])
        num_link_features = len(features["links"].keys())
        link_features = torch.zeros((n_links, num_link_features))

        for i in range(n_links):
            link_features[i][0] = features["links"]['l_link_load'][i]
            link_features[i][1] = features["links"]['l_link_load2'][i]
            link_features[i][2] = features["links"]['l_link_load3'][i]

        assert num_link_features == 3

        return link_features

    def _get_path_features(self, features):
        """
        This will return a matrix / 2d array of the shape
        [Number of paths, Path feature size]
        """
        n_paths = len(features["paths"][next(iter(features["paths"]))])

        # because of the label
        num_path_features = len(features["paths"].keys()) - 1
        path_features = torch.zeros((n_paths, num_path_features))

        for i in range(n_paths):
            path_features[i][0] = features['paths']['p_AvgBw'][i]
            path_features[i][1] = features['paths']['p_PktsGen'][i]
            path_features[i][2] = features['paths']['p_time_EqLambda'][i]

        assert num_path_features == 3

        return path_features

    def _get_label(self, features):
        return features["paths"]['p_AvgDelay']

    def len(self):
        if self.test:
            return 1_560
        elif self.val:
            return 3_120
        return 120_000

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(
                self.processed_dir, f'data_test_{idx}.pt'))
        elif self.val:
            data = torch.load(os.path.join(
                self.processed_dir, f'data_val_{idx}.pt'))
        else:
            data = torch.load(os.path.join(
                self.processed_dir, f'data_{idx}.pt'))
        return data


"""
for G,R,P,D in generator('./data/raw/gnnet-ch21-dataset-train'):
    G = nx.DiGraph(G)
    print(P[24][1]['qosQueuesStats'][0]['avgPortOccupancy']*32/G.edges[24,1]['bandwidth']+P[0][24]['qosQueuesStats'][0]['avgPortOccupancy']*32/G.edges[0, 24]['bandwidth'])
    print(R[0][1]) # 0,24,1
    print(D[0, 1]['Flows'][0]['AvgDelay'])
    break
"""
'''
for x in generator('./data/raw/gnnet-ch21-dataset-train',stats=True):
    features, index = networkx_to_data(x)
    # print(features)
    #print(networkx_to_data(x))
    for node in x.nodes(data=True):
        #if node[0].split('_')[0]=="l":
        #print(node)
        pass

    #print(x)
    break
'''

if __name__ == '__main__':
    # accum_stats shoulf be true and get_all should be true
    train_dataset = GNNC21Dataset(
        root='data/', filename='gnnet-ch21-dataset-train', stats=True)
    val_dataset = GNNC21Dataset(
        root='data/', filename='gnnet-ch21-dataset-validation', val=True, stats=True, accum_stats=True)
    test_dataset = GNNC21Dataset(
        root='data/', filename='gnnet-ch21-dataset-test-with-labels', test=True, stats=True, accum_stats=True)

'''
with open(r"train", "rb") as input_file:
    e = pickle.load(input_file)
    print(e)
'''
