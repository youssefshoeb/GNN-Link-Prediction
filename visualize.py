import datanetAPI
import torch
import tqdm
import numpy as np
import networkx as nx


def generator(data_dir, intensity_values=[], topology_sizes=[], shuffle=False):
    """
    This function uses the DatanetAPI to output a single sample of the data.
    """
    tool = datanetAPI.Datanet21API(
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
        yield networkx_to_data(HG)


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


def input_to_networkx(network_graph, routing_matrix, traffic_matrix, performance_matrix, port_stats):
    """
    This function converts a dataset sample to a networkx line graph
    """
    G = nx.DiGraph(network_graph)
    R = routing_matrix
    T = traffic_matrix
    D = performance_matrix
    P = port_stats

    # Create a directed Networkx object
    D_G = nx.DiGraph()
    O_G = nx.DiGraph()
    I_G = nx.DiGraph()

    # Add all nodes to the graph
    for src in range(G.number_of_nodes()):
        src_feats = {"node_feature_1": 1, "node_feature_2": 1, "node_feature_3": 1, "Type": 'node'}
        D_G.add_node('n_{}'.format(src))
        O_G.add_node('n_{}'.format(src))
        I_G.add_node('n_{}'.format(src), **src_feats)

    # Iterate over all nodes in the graph
    for src in range(G.number_of_nodes()):
        for dst in range(G.number_of_nodes()):
            if src != dst:
                # Add links to the graph as nodes
                if G.has_edge(src, dst):
                    # Add source node stats to link features
                    src_feats = dict((f'n_{k}', v)
                                     for k, v in G.nodes[src].items())
                    # link_feats = dict((f'l_{k}', v)
                    #                  for k, v in G.edges[src, dst].items())
                    link_feats = {'l_bandwidth': G.edges[src, dst]['bandwidth']}

                    # link_feats.update(src_feats)
                    # For visualization
                    link_feats.update({'Type': 'link'})

                    D_G.add_node('l_{}_{}'.format(src, dst), **link_feats)
                    I_G.add_node('l_{}_{}'.format(src, dst), **link_feats)

                    # Connect each link to its source and destination node
                    D_G.add_edge('n_{}'.format(
                        src), 'l_{}_{}'.format(src, dst))
                    D_G.add_edge('l_{}_{}'.format(
                        src, dst), 'n_{}'.format(dst))

                    I_G.add_edge('n_{}'.format(
                        src), 'l_{}_{}'.format(src, dst), edge_type='n-l')
                    I_G.add_edge('l_{}_{}'.format(
                        src, dst), 'n_{}'.format(dst), edge_type='l-n')

                    O_G.add_edge('n_{}'.format(src), 'n_{}'.format(dst), **link_feats)

                # Add paths to the graph as nodes
                for f_id in range(len(T[src, dst]['Flows'])):
                    if T[src, dst]['Flows'][f_id]['AvgBw'] != 0 and T[src, dst]['Flows'][f_id]['PktsGen'] != 0:
                        dct_flows = dict(
                            (f'p_{k}', v) for k, v in T[src, dst]['Flows'][f_id].items())
                        # for visualizeation remove enums
                        dct_flows.pop('p_TimeDist')
                        dct_flows.pop('p_SizeDist')
                        dct_flows.update({'p_TimeDist': T[src, dst]['Flows'][f_id]['TimeDist'].name})
                        dct_flows.update({'p_SizeDist': T[src, dst]['Flows'][f_id]['SizeDist'].name})
                        dct_flows.update({'Type': 'path'})
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
                        I_G.add_node('p_{}_{}_{}'.format(
                            src, dst, f_id), **dct_flows)

                        # Add edges between paths and all links that traverse the path
                        for _, (h_1, h_2) in enumerate([R[src, dst][i:i + 2] for i in range(0, len(R[src, dst]) - 1)]):
                            _p = 'p_{}_{}_{}'.format(src, dst, f_id)
                            _l = 'l_{}_{}'.format(h_1, h_2)
                            _n1 = 'n_{}'.format(h_1)
                            _n2 = 'n_{}'.format(h_2)

                            if _n1 not in I_G[_p]:
                                I_G.add_edge(_p, _n1, edge_type='p-n')
                                I_G.add_edge(_n1, _p, edge_type='n-p')

                            if _n2 not in I_G[_p]:
                                I_G.add_edge(_p, _n2, edge_type='p-n')
                                I_G.add_edge(_n1, _p, edge_type='n-p')

                            D_G.add_edge(_p, _l, edge_type='p-l')
                            D_G.add_edge(_l, _p, edge_type='l-p')
                            I_G.add_edge(_p, _l, edge_type='p-l')
                            I_G.add_edge(_l, _p, edge_type='l-p')

    # Add edges between links connected by a node.
    edge_list = list(D_G.edges())
    for src, dst in edge_list:
        if src.split('_')[0] == 'l':
            for _, dst2 in D_G.out_edges(dst):
                edge = (src, dst2)
                if edge not in D_G.edges(src) and dst2 != src:
                    D_G.add_edge(src, dst2, edge_type='l-l')

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

            link_loads[node] = {
                "l_link_load": load, "l_link_load2": load * load, "l_link_load3": load * load * load}

    nx.set_node_attributes(D_G, link_loads)
    nx.write_gexf(O_G, "trainset_example_input-original.gexf")
    nx.write_gexf(I_G, "trainset_example_input-inter.gexf")
    nx.write_gexf(D_G, "trainset_example_input-modified.gexf")
    return D_G


# for features, index in generator('./data/GNN-CH21/raw/gnnet-ch21-dataset-test-with-labels', topology_sizes=[300]):  # topology_sizes=[300]):
for features, index in generator('./data/GNN-CH21/raw/gnnet-ch21-dataset-train'):
    # print(features)
    # print(networkx_to_data(x))
    # for node in x.nodes(data=True):
    #    if node[0].split('_')[0]=="l":
    #    print(node)
    #    pass

    # print(x)
    break
