import os
import torch
import torch_geometric
import datanetAPI
import multiprocessing

import networkx as nx
import numpy as np
from tqdm import tqdm


RAW_DIRS = {'train': './dataset/gnnet-ch21-dataset-train',
            'validation': './dataset/gnnet-ch21-dataset-validation',
            'test': './dataset/gnnet-ch21-dataset-test-with-labels'
            }

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

    # Remove any nodes that have an out degree of zero
    D_G.remove_nodes_from([node for node, out_degree in D_G.degree() if out_degree == 0])
    
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


if __name__ == "__main__":
    # Generate torch files
    generate_files()

    
