from msilib.schema import Error
import datanetAPI
import os
import torch
import torch_geometric
import tqdm
import numpy as np
import networkx as nx


def generator(data_dir, dataset, intensity_values=[], topology_sizes=[], shuffle=False):
    """
    This function uses the DatanetAPI to output a single sample of the data.
    """
    if dataset == "GNNCH21":
        tool = datanetAPI.Datanet21API(
            data_dir, intensity_values, topology_sizes, shuffle=shuffle)
    elif dataset == "GNNCH20":
        tool = datanetAPI.Datanet20API(
            data_dir, intensity_values, shuffle=shuffle)
    else:
        raise Error("Only 'GNNCH21' and 'GNNCH20' are currently supported")

    it = iter(tool)
    num_samples = 0

    for sample in tqdm.tqdm(it):
        G_copy = sample.get_topology_object().copy()
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        D = sample.get_performance_matrix()
        HG = input_to_networkx(network_graph=G_copy,
                               routing_matrix=R,
                               traffic_matrix=T,
                               performance_matrix=D)
        num_samples += 1
        yield networkx_to_data(HG)


def hetero_generator(data_dir, dataset, intensity_values=[], topology_sizes=[], shuffle=False):
    """
    This function uses the DatanetAPI to output a single sample of the data.
    """
    if dataset == "GNNCH21":
        tool = datanetAPI.Datanet21API(
            data_dir, intensity_values, topology_sizes, shuffle=shuffle)
    elif dataset == "GNNCH20":
        tool = datanetAPI.Datanet20API(
            data_dir, intensity_values, shuffle=shuffle)

    it = iter(tool)
    num_samples = 0

    for sample in tqdm.tqdm(it):
        G_copy = sample.get_topology_object().copy()
        T = sample.get_traffic_matrix()
        R = sample.get_routing_matrix()
        D = sample.get_performance_matrix()
        HG = input_to_hetero_networkx(network_graph=G_copy,
                                      routing_matrix=R,
                                      traffic_matrix=T,
                                      performance_matrix=D)
        num_samples += 1
        yield hetero_networkx_to_data(HG)


def input_to_networkx(network_graph, routing_matrix, traffic_matrix, performance_matrix):
    """
    This function converts a dataset sample to a networkx line graph.
    To be used for GNN-CH21 Dataset.
    Uncomment lines to return back to more generic mode that allows for
    saving all featurs (Warning: Extremly Slow)
    """
    G = nx.DiGraph(network_graph)
    R = routing_matrix
    T = traffic_matrix
    D = performance_matrix

    # Create a directed Networkx object
    D_G = nx.DiGraph()

    # Add all nodes to the graph
    for src in range(G.number_of_nodes()):
        D_G.add_node('n_{}'.format(src))

    # Iterate over all nodes in the graph
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                # Add links to the graph as nodes
                if G.has_edge(src, dst):
                    # Add source node stats to link features
                    # src_feats = dict((f'n_{k}', v)
                    #                 for k, v in G.nodes[src].items())
                    # link_feats = dict((f'n_{k}', v)
                    #                 for k, v in G.edges[src, dst].items())
                    link_feats = {'l_bandwidth': G.edges[src, dst]['bandwidth']}

                    # if 'n_schedulingWeights' not in src_feats:
                    #    src_feats.update({'n_schedulingWeights': '100,100,100'})

                    # link_feats.update(src_feats)
                    # For visualization
                    # link_feats.update({'Type': 'link'})

                    D_G.add_node('l_{}_{}'.format(src, dst), **link_feats)

                    # Connect each link to its source and destination node
                    D_G.add_edge('n_{}'.format(
                        src), 'l_{}_{}'.format(src, dst))
                    D_G.add_edge('l_{}_{}'.format(
                        src, dst), 'n_{}'.format(dst))

                # Add paths to the graph as nodes
                for f_id in range(len(T[src, dst]['Flows'])):
                    if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                        dct_flows = dict(
                            (f'p_{k}', v) for k, v in T[src, dst]['Flows'][f_id].items())

                        # For visualizeation remove enums
                        dct_flows.pop('p_TimeDist')
                        dct_flows.pop('p_SizeDist')
                        dct_flows.update({'p_TimeDist': T[src, dst]['Flows'][f_id]['TimeDist'].name})
                        dct_flows.update({'p_SizeDist': T[src, dst]['Flows'][f_id]['SizeDist'].name})
                        # dct_flows.update({'Type': 'path'})
                        # Flatten dict of dicts to just a dict
                        dct_flows.pop('p_SizeDistParams')
                        dct_flows.pop('p_TimeDistParams')
                        dct_flows_size = dict(
                            (f'p_size_{k}', v) for k, v in T[src, dst]['Flows'][f_id]['SizeDistParams'].items())
                        dct_flows_time = dict(
                            (f'p_time_{k}', v) for k, v in T[src, dst]['Flows'][f_id]['TimeDistParams'].items())
                        dct_flows.update(dct_flows_size)
                        dct_flows.update(dct_flows_time)
                        dct_flows['p_AvgDelay'] = D[src,
                                                    dst]['Flows'][f_id]['AvgDelay']

                        # remove some features we already know have no effect
                        dct_flows.pop('p_ToS')
                        dct_flows.pop('p_time_ExpMaxFactor')
                        dct_flows.pop('p_TimeDist')
                        dct_flows.pop('p_SizeDist')
                        dct_flows.pop('p_size_AvgPktSize')
                        dct_flows.pop('p_size_PktSize1')
                        dct_flows.pop('p_size_PktSize2')
                        dct_flows.pop('p_TotalPktsGen')

                        # divide features by p_time_AvgPktsLambda
                        dct_flows['p_PktsGen*'] = dct_flows['p_PktsGen'] / \
                            dct_flows['p_time_AvgPktsLambda']
                        dct_flows['p_AvgBw*'] = dct_flows['p_AvgBw'] / \
                            dct_flows['p_time_AvgPktsLambda']
                        # dct_flows['p_time_EqLambda*'] = dct_flows['p_time_EqLambda'] / \
                        #    dct_flows['p_time_AvgPktsLambda']

                        dct_flows.pop('p_time_AvgPktsLambda')
                        dct_flows.pop('p_PktsGen')

                        D_G.add_node('p_{}_{}_{}'.format(
                            src, dst, f_id), **dct_flows)

                        # Add edges between paths and all links that traverse the path
                        for _, (h_1, h_2) in enumerate([R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]):
                            _p = 'p_{}_{}_{}'.format(src, dst, f_id)
                            _l = 'l_{}_{}'.format(h_1, h_2)
                            D_G.add_edge(_p, _l)
                            D_G.add_edge(_l, _p)

    # Add edges between links connected by a node.
    edge_list = list(D_G.edges())
    for src, dst in edge_list:
        if src.split('_')[0] == 'l':
            for _, dst2 in D_G.out_edges(dst):
                edge = (src, dst2)
                if edge not in D_G.edges(src) and dst2 != src:
                    D_G.add_edge(src, dst2)

    # Remove nodes (Conversion to Line Graph)
    for src in list(D_G.nodes):
        if src.split('_')[0] == 'n':
            D_G.remove_node(src)

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

    nx.set_node_attributes(D_G, link_loads)

    return D_G


def input_to_hetero_networkx(network_graph, routing_matrix, traffic_matrix, performance_matrix):
    """
    This function converts a dataset sample to a networkx line graph.
    To be used for GNN-CH21 Dataset.
    Uncomment lines to return back to more generic mode that allows for
    saving all featurs (Warning: Extremly Slow)
    """
    G = nx.DiGraph(network_graph)
    R = routing_matrix
    T = traffic_matrix
    D = performance_matrix

    # Create a directed Networkx object
    D_G = nx.DiGraph()

    # Add all nodes to the graph
    for src in range(G.number_of_nodes()):
        src_feats = {"node_feature_1": 1, "node_feature_2": 1, "node_feature_3": 1}
        D_G.add_node('n_{}'.format(src), **src_feats)

    # Iterate over all nodes in the graph
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                # Add links to the graph as nodes
                if G.has_edge(src, dst):
                    # Add source node stats to link features
                    # src_feats = dict((f'n_{k}', v)
                    #                 for k, v in G.nodes[src].items())
                    # print(src_feats)
                    # link_feats = dict((f'n_{k}', v)
                    #                 for k, v in G.edges[src, dst].items())
                    link_feats = {'l_bandwidth': G.edges[src, dst]['bandwidth']}

                    # if 'n_schedulingWeights' not in src_feats:
                    #    src_feats.update({'n_schedulingWeights': '100,100,100'})

                    # link_feats.update(src_feats)
                    # For visualization
                    # link_feats.update({'Type': 'link'})

                    D_G.add_node('l_{}_{}'.format(src, dst), **link_feats)

                    # Connect each link to its source and destination node
                    D_G.add_edge('n_{}'.format(
                        src), 'l_{}_{}'.format(src, dst))
                    D_G.add_edge('l_{}_{}'.format(
                        src, dst), 'n_{}'.format(dst))

                # Add paths to the graph as nodes
                for f_id in range(len(T[src, dst]['Flows'])):
                    if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                        dct_flows = dict(
                            (f'p_{k}', v) for k, v in T[src, dst]['Flows'][f_id].items())

                        # For visualizeation remove enums
                        dct_flows.pop('p_TimeDist')
                        dct_flows.pop('p_SizeDist')
                        dct_flows.update({'p_TimeDist': T[src, dst]['Flows'][f_id]['TimeDist'].name})
                        dct_flows.update({'p_SizeDist': T[src, dst]['Flows'][f_id]['SizeDist'].name})
                        # dct_flows.update({'Type': 'path'})
                        # Flatten dict of dicts to just a dict
                        dct_flows.pop('p_SizeDistParams')
                        dct_flows.pop('p_TimeDistParams')
                        dct_flows_size = dict(
                            (f'p_size_{k}', v) for k, v in T[src, dst]['Flows'][f_id]['SizeDistParams'].items())
                        dct_flows_time = dict(
                            (f'p_time_{k}', v) for k, v in T[src, dst]['Flows'][f_id]['TimeDistParams'].items())
                        dct_flows.update(dct_flows_size)
                        dct_flows.update(dct_flows_time)
                        dct_flows['p_AvgDelay'] = D[src,
                                                    dst]['Flows'][f_id]['AvgDelay']

                        # remove some features we already know have no effect
                        dct_flows.pop('p_ToS')
                        dct_flows.pop('p_time_ExpMaxFactor')
                        dct_flows.pop('p_TimeDist')
                        dct_flows.pop('p_SizeDist')
                        dct_flows.pop('p_size_AvgPktSize')
                        dct_flows.pop('p_size_PktSize1')
                        dct_flows.pop('p_size_PktSize2')
                        dct_flows.pop('p_TotalPktsGen')

                        # divide features by p_time_AvgPktsLambda
                        dct_flows['p_PktsGen*'] = dct_flows['p_PktsGen'] / \
                            dct_flows['p_time_AvgPktsLambda']
                        dct_flows['p_AvgBw*'] = dct_flows['p_AvgBw'] / \
                            dct_flows['p_time_AvgPktsLambda']
                        # dct_flows['p_time_EqLambda*'] = dct_flows['p_time_EqLambda'] / \
                        #    dct_flows['p_time_AvgPktsLambda']

                        dct_flows.pop('p_time_AvgPktsLambda')
                        dct_flows.pop('p_PktsGen')

                        D_G.add_node('p_{}_{}_{}'.format(
                            src, dst, f_id), **dct_flows)

                        # Add edges between paths and all links that traverse the path
                        for _, (h_1, h_2) in enumerate([R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]):
                            _p = 'p_{}_{}_{}'.format(src, dst, f_id)
                            _l = 'l_{}_{}'.format(h_1, h_2)
                            _n1 = 'n_{}'.format(h_1)
                            _n2 = 'n_{}'.format(h_2)

                            # if _n1 not in D_G[_p]:
                            #    D_G.add_edge(_p, _n1)
                            #    D_G.add_edge(_n1, _p)

                            # if _n2 not in D_G[_p]:
                            #    D_G.add_edge(_p, _n2)
                            #    D_G.add_edge(_n1, _p)

                            D_G.add_edge(_p, _l)
                            D_G.add_edge(_l, _p)

    # Add edges between links connected by a node.
    # edge_list = list(D_G.edges())
    # for src, dst in edge_list:
    #    if src.split('_')[0] == 'l':
    #        for _, dst2 in D_G.out_edges(dst):
    #            edge = (src, dst2)
    #            if edge not in D_G.edges(src) and dst2 != src:
    #                D_G.add_edge(src, dst2)

    # Remove nodes (Conversion to Line Graph)
    # for src in list(D_G.nodes):
    #    if src.split('_')[0] == 'n':
    #        D_G.remove_node(src)

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

    nx.set_node_attributes(D_G, link_loads)

    return D_G


def networkx_to_data(G):
    """
    Convert the networkx line graph to a feature matrix and edge index.
    We save the path and link index to correctly get the edge indicies
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


def hetero_networkx_to_data(G):
    """
    Convert the networkx graph to a feature matrix and edge index.
    We save the path and link index to correctly get the edge indicies
    """
    feature_dict = {}
    key_dict = {}
    paths_i = 0
    links_i = 0
    nodes_i = 0

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

        elif key.split('_')[0] == 'n':
            key_dict[key] = nodes_i

            if nodes_i == 0:
                feature_dict["nodes"] = dict(
                    (k, np.array([])) for k, _ in feat_dict.items())

            node_features = dict((k, v) for k, v in feat_dict.items())

            for k, v in feature_dict["nodes"].items():
                feature_dict["nodes"][k] = np.append([v], [node_features[k]])

            nodes_i += 1

    # Get edges index
    edge_dict = {"p-l": [], "l-p": [], "l-n": [], "n-l": []}
    for edge in G.edges:
        if edge[0].split("_")[0] == 'p' and edge[1].split("_")[0] == 'l':
            edge_dict["p-l"].append([key_dict[edge[0]], key_dict[edge[1]]])

        elif edge[0].split("_")[0] == 'l' and edge[1].split("_")[0] == 'p':
            edge_dict["l-p"].append([key_dict[edge[0]], key_dict[edge[1]]])

        elif edge[0].split("_")[0] == 'l' and edge[1].split("_")[0] == 'n':
            edge_dict["l-n"].append([key_dict[edge[0]], key_dict[edge[1]]])

        elif edge[0].split("_")[0] == 'n' and edge[1].split("_")[0] == 'l':
            edge_dict["n-l"].append([key_dict[edge[0]], key_dict[edge[1]]])

    edge_dict["p-l"] = torch.tensor(edge_dict["p-l"]).t().contiguous()
    edge_dict["l-p"] = torch.tensor(edge_dict["l-p"]).t().contiguous()
    edge_dict["l-n"] = torch.tensor(edge_dict["l-n"]).t().contiguous()
    edge_dict["n-l"] = torch.tensor(edge_dict["n-l"]).t().contiguous()

    return feature_dict, edge_dict


class TorchDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename, dataset_name, intensity=[], topology_sizes=[], shuffle=False):
        'Initialization'
        self.generator_object = generator(
            filename, dataset_name, intensity, topology_sizes, shuffle=shuffle)

    def __iter__(self):
        'Generates one sample of data'
        return iter(self.generator_object)


class HETROTorchDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename, dataset_name, intensity=[], topology_sizes=[], shuffle=False):
        'Initialization'
        self.generator_object = hetero_generator(
            filename, dataset_name, intensity, topology_sizes, shuffle=shuffle)

    def __iter__(self):
        'Generates one sample of data'
        return iter(self.generator_object)


class GNNC21Dataset(torch_geometric.data.Dataset):
    def __init__(self, root, filename, val=False, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.val = val
        self.filename = filename
        self.dataset_name = "GNNCH21"

        super(GNNC21Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        # Add the name of the dataset gnnet_data_set_training
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

    def transformation(self, x):
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
        x["paths"]['p_PktsGen*'] = (
            x["paths"]['p_PktsGen*'] - 0.7498462848223999) / (1.3038396633263194 - 0.7498462848223999)
        x["paths"]['p_AvgBw*'] = (
            x["paths"]['p_AvgBw*'] - 622.2114877920252) / (1346.7849033971643 - 622.2114877920252)
        x["paths"]['p_time_EqLambda'] = (
            x["paths"]['p_time_EqLambda'] - 40.0337) / (1999.52 - 40.0337)
        x["paths"]['p_AvgDelay'] = (
            x["paths"]['p_AvgDelay'] - 0.000418) / (9.15503 - 0.000418)

        return x

    def download(self):
        raise NotImplementedError("Download function not implemented")

    def process(self):
        dataset = TorchDataset(os.path.join(self.raw_dir, self.filename), self.dataset_name)
        index = 0

        for feature_dict, edge_dict in dataset:
            feature_dict = self.transformation(feature_dict)

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

            # Label
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

    def _get_link_features(self, features):
        """
        This will return a matrix / 2d array of the shape
        [Number of links, link Feature size]
        """
        """
        Features ignored: 'l_bandwidth'
        """
        n_links = len(features["links"][next(iter(features["links"]))])
        num_link_features = 3
        link_features = torch.zeros((n_links, num_link_features))

        for i in range(n_links):
            link_features[i][0] = features["links"]['l_link_load'][i]
            link_features[i][1] = features["links"]['l_link_load2'][i]
            link_features[i][2] = features["links"]['l_link_load3'][i]

        return link_features

    def _get_path_features(self, features):
        """
        This will return a matrix / 2d array of the shape
        [Number of paths, Path feature size]
        """
        """
        Features ignored: 'p_ToS', 'p_time_ExpMaxFactor', 'p_TimeDist', 'p_SizeDist',
        'p_size_AvgPktSize', 'p_size_PktSize1', 'p_size_PktSize2', 'p_TotalPktsGen',
        'p_time_AvgPktsLambda
        """
        n_paths = len(features["paths"][next(iter(features["paths"]))])
        num_path_features = 3
        path_features = torch.zeros((n_paths, num_path_features))

        for i in range(n_paths):
            path_features[i][0] = features['paths']['p_AvgBw*'][i]
            path_features[i][1] = features['paths']['p_PktsGen*'][i]
            path_features[i][2] = features['paths']['p_time_EqLambda'][i]

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


class HETROGNNC21Dataset(torch_geometric.data.Dataset):
    def __init__(self, root, filename, val=False, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.val = val
        self.filename = filename
        self.dataset_name = "GNNCH21"

        super(HETROGNNC21Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        # Add the name of the dataset gnnet_data_set_training
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

    def transformation(self, x):
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
        x["paths"]['p_PktsGen*'] = (
            x["paths"]['p_PktsGen*'] - 0.7498462848223999) / (1.3038396633263194 - 0.7498462848223999)
        x["paths"]['p_AvgBw*'] = (
            x["paths"]['p_AvgBw*'] - 622.2114877920252) / (1346.7849033971643 - 622.2114877920252)
        x["paths"]['p_time_EqLambda'] = (
            x["paths"]['p_time_EqLambda'] - 40.0337) / (1999.52 - 40.0337)
        x["paths"]['p_AvgDelay'] = (
            x["paths"]['p_AvgDelay'] - 0.000418) / (9.15503 - 0.000418)

        return x

    def download(self):
        raise NotImplementedError("Download function not implemented")

    def process(self):
        dataset = HETROTorchDataset(os.path.join(self.raw_dir, self.filename), self.dataset_name)
        index = 0

        for feature_dict, edge_dict in dataset:
            feature_dict = self.transformation(feature_dict)

            # Create data object
            data = torch_geometric.data.HeteroData()

            # Get node features and labels
            data['node'].x = self._get_node_features(
                feature_dict)  # [num_link, num_features_link]
            data['link'].x = self._get_link_features(
                feature_dict)  # [num_link, num_features_link]
            data['path'].x = self._get_path_features(
                feature_dict)  # [num_path, num_features_path]

            # Get adjacency info
            data['path', 'uses', 'link'].edge_index = edge_dict['p-l']
            data['link', 'includes', 'path'].edge_index = edge_dict['l-p']
            data['link', 'connects', 'node'].edge_index = edge_dict['l-n']
            data['node', 'has', 'link'].edge_index = edge_dict['n-l']
            # data['node', 'is_included', 'path'].edge_index = edge_dict['n-p']
            # data['path', 'connects', 'node'].edge_index = edge_dict['p-n']

            # Label
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

    def _get_link_features(self, features):
        """
        This will return a matrix / 2d array of the shape
        [Number of links, link Feature size]
        """
        """
        Features ignored: 'l_bandwidth'
        """
        n_links = len(features["links"][next(iter(features["links"]))])
        num_link_features = 3
        link_features = torch.zeros((n_links, num_link_features))

        for i in range(n_links):
            link_features[i][0] = features["links"]['l_link_load'][i]
            link_features[i][1] = features["links"]['l_link_load2'][i]
            link_features[i][2] = features["links"]['l_link_load3'][i]

        return link_features

    def _get_path_features(self, features):
        """
        This will return a matrix / 2d array of the shape
        [Number of paths, Path feature size]
        """
        """
        Features ignored: 'p_ToS', 'p_time_ExpMaxFactor', 'p_TimeDist', 'p_SizeDist',
        'p_size_AvgPktSize', 'p_size_PktSize1', 'p_size_PktSize2', 'p_TotalPktsGen',
        'p_time_AvgPktsLambda
        """
        n_paths = len(features["paths"][next(iter(features["paths"]))])
        num_path_features = 3
        path_features = torch.zeros((n_paths, num_path_features))

        for i in range(n_paths):
            path_features[i][0] = features['paths']['p_AvgBw*'][i]
            path_features[i][1] = features['paths']['p_PktsGen*'][i]
            path_features[i][2] = features['paths']['p_time_EqLambda'][i]

        return path_features

    def _get_node_features(self, features):
        """
        This will return a matrix / 2d array of the shape
        [Number of paths, Path feature size]
        """
        n_paths = len(features["nodes"][next(iter(features["nodes"]))])
        num_path_features = 3
        node_features = torch.zeros((n_paths, num_path_features))

        for i in range(n_paths):
            node_features[i][0] = features['nodes']['node_feature_1'][i]
            node_features[i][1] = features['nodes']['node_feature_2'][i]
            node_features[i][2] = features['nodes']['node_feature_3'][i]

        return node_features

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


class GNNC20Dataset(torch_geometric.data.Dataset):
    def __init__(self, root, filename, val=False, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.val = val
        self.filename = filename
        self.dataset_name = "GNNCH20"

        super(GNNC20Dataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        # Add the name of the dataset gnnet_data_set_training
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

    def transformation(self, x):
        """Apply a transformation over all the samples included in the dataset.
            Args:
                x (dict): predictor variable.
            Returns:
                x,y: The modified predictor/target variables.
            """
        x["links"]['l_link_load'] = (
            x["links"]['l_link_load'] - 0.0) / (4.203 - 0.001)
        x["links"]['l_link_load2'] = (
            x["links"]['l_link_load2']) / (17.663 - 0.0)
        x["links"]['l_link_load3'] = (
            x["links"]['l_link_load3']) / (74.234 - 0.0)
        x["paths"]['p_PktsGen*'] = (
            x["paths"]['p_PktsGen*'] - 0.899) / (1.102 - 0.899)
        x["paths"]['p_AvgBw*'] = (
            x["paths"]['p_AvgBw*'] - 881.034) / (1134.054 - 881.034)
        x["paths"]['p_time_EqLambda'] = (
            x["paths"]['p_time_EqLambda'] - 40.0237) / (1999.63 - 40.0237)
        x["paths"]['p_AvgDelay'] = (
            x["paths"]['p_AvgDelay'] - 0.0075) / (246006.0 - 0.0075)

        schedulingweights = [y.split(',') for y in x["links"]['n_schedulingWeights']]
        x["links"]['n_schedulingWeights'] = [f'{float(elem[0])/100},{float(elem[1])/100},{float(elem[2])/100}' for elem in schedulingweights]

        return x

    def download(self):
        raise NotImplementedError("Download function not implemented")

    def process(self):
        dataset = TorchDataset(os.path.join(self.raw_dir, self.filename), self.dataset_name)
        index = 0

        for feature_dict, edge_dict in dataset:
            feature_dict = self.transformation(feature_dict)

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

            # Label
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

    def _get_link_features(self, features):
        """
        This will return a matrix / 2d array of the shape
        [Number of links, link Feature size]
        """
        """
        Features ignored: 'l_bandwidth'
        """
        n_links = len(features["links"][next(iter(features["links"]))])
        num_link_features = 9
        link_features = torch.zeros((n_links, num_link_features))

        for i in range(n_links):
            link_features[i][0] = features["links"]['l_link_load'][i]
            link_features[i][1] = features["links"]['l_link_load2'][i]
            link_features[i][2] = features["links"]['l_link_load3'][i]
            policy = features["links"]['n_schedulingPolicy'][i]
            if policy == "WFQ":
                link_features[i][3] = 1
                link_features[i][4] = 0
                link_features[i][5] = 0
            elif policy == "SP":
                link_features[i][3] = 0
                link_features[i][4] = 1
                link_features[i][5] = 0
            elif policy == "DRR":
                link_features[i][3] = 0
                link_features[i][4] = 0
                link_features[i][5] = 1
            weights = features["links"]['n_schedulingWeights'][i].split(',')
            link_features[i][6] = float(weights[0])
            link_features[i][7] = float(weights[1])
            link_features[i][8] = float(weights[2])

        return link_features

    def _get_path_features(self, features):
        """
        This will return a matrix / 2d array of the shape
        [Number of paths, Path feature size]
        """
        """
        Features ignored: 'p_ToS', 'p_time_ExpMaxFactor', 'p_TimeDist', 'p_SizeDist',
        'p_size_AvgPktSize', 'p_size_PktSize1', 'p_size_PktSize2', 'p_TotalPktsGen',
        'p_time_AvgPktsLambda
        """
        n_paths = len(features["paths"][next(iter(features["paths"]))])
        num_path_features = 6
        path_features = torch.zeros((n_paths, num_path_features))

        for i in range(n_paths):
            path_features[i][0] = features['paths']['p_AvgBw*'][i]
            path_features[i][1] = features['paths']['p_PktsGen*'][i]
            path_features[i][2] = features['paths']['p_time_EqLambda'][i]
            tos = features['paths']['p_ToS'][i]
            if tos == 0:
                path_features[i][3] = 1
                path_features[i][4] = 0
                path_features[i][5] = 0
            elif tos == 1:
                path_features[i][3] = 0
                path_features[i][4] = 1
                path_features[i][5] = 0
            elif tos == 2:
                path_features[i][3] = 0
                path_features[i][4] = 0
                path_features[i][5] = 1

        return path_features

    def _get_label(self, features):
        return features["paths"]['p_AvgDelay']

    def len(self):
        if self.test:
            return 50_000
        elif self.val:
            return 100_000
        return 400_000

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
"""
for features, index in generator('./data/GNN-CH21/raw/gnnet-ch21-dataset-train', dataset="GNNCH21"):
    print(features)
    # print(networkx_to_data(x))
    # for node in x.nodes(data=True):
        # if node[0].split('_')[0]=="l":
        # print(node)
    #    pass

    # print(x)
    break
"""

if __name__ == '__main__':
    train_dataset = GNNC21Dataset(
        root='data/GNNCH21/', filename='gnnet-ch21-dataset-train')
    val_dataset = GNNC21Dataset(
        root='data/GNNCH21/', filename='gnnet-ch21-dataset-validation', val=True)
    test_dataset = GNNC21Dataset(
        root='data/GNNCH21/', filename='gnnet-ch21-dataset-test-with-labels', test=True)
    """
    train_dataset = GNNC20Dataset(
        root='data/GNNCH20/', filename='gnnet_data_set_training')
    val_dataset = GNNC20Dataset(
        root='data/GNNCH20/', filename='gnnet_data_set_validation', val=True)
    test_dataset = GNNC20Dataset(
        root='data/GNNCH20/', filename='gnnet_data_set_evaluation_delays', test=True)
    """
