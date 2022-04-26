import torch
import torch_geometric
import torch_scatter

from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.conv.hetero_conv import HeteroConv

from torch_geometric.utils import to_dense_adj
from typing import Callable, Any, Union
from torch.functional import Tensor
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_sparse import SparseTensor, matmul


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
        blocking_probs = 0.5 * torch.ones_like(A)

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
                T += torch_scatter.scatter(src=torch.gather(traffic, 0, which_paths),
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
        res = torch_scatter.scatter(src=E, index=edges_pl[0, :], dim=0, dim_size=X.size(0), reduce='sum')
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
    def __init__(self, input_channels: dict, node_embedding_size: int, message_passing_layers: int, dropout: float, concat_path: bool, bl_features: bool, divided_features:bool, global_feats: bool,
                 mlp_layers: list, act, mlp_head_act, mlp_bn: bool):
        super().__init__()
        self.num_layers = message_passing_layers
        self.concat_path = concat_path
        self.bl_features = bl_features
        self.divided_features = divided_features
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.global_feats = global_feats

        if not self.divided_features:
            input_channels['path'] = input_channels['path'] - 3
            input_channels['link'] = input_channels['link'] - 1
            if not self.bl_features:
                input_channels['path'] = input_channels['path'] - 1
                input_channels['link'] = input_channels['link'] - 3
        else:
            if not self.bl_features:
                input_channels['path'] = input_channels['path'] - 1
                input_channels['link'] = input_channels['link'] - 3

        if global_feats:
            self.global_feats_size = 8
        else:
            self.global_feats_size = 0

        if concat_path:
            self.concat_size = input_channels['path']
        else:
            self.concat_size = 0

        self.convs = torch.nn.ModuleList()
        self.readout = torch.nn.ModuleList()

        # Message Passing
        # first conv layer
        self.convs.append(HeteroConv({
            ('path', 'uses', 'link'): GINLayer(input_channels['path'] + input_channels['link'], node_embedding_size, concat=True),
            ('link', 'includes', 'path'): GINLayer(input_channels['link'] + input_channels['path'], node_embedding_size, concat=True),
            ('link', 'connects', 'node'): GINLayer(input_channels['link'] + input_channels['node'], node_embedding_size, concat=True),
            ('node', 'has', 'link'): GINLayer(input_channels['node'] + input_channels['link'], node_embedding_size, concat=True)}, aggr='sum'))

        # remaining conv layers
        for _ in range(self.num_layers - 1):
            self.convs.append(HeteroConv({
                ('path', 'uses', 'link'): GINLayer(node_embedding_size, node_embedding_size),
                ('link', 'includes', 'path'): GINLayer(node_embedding_size, node_embedding_size),
                ('link', 'connects', 'node'): GINLayer(node_embedding_size, node_embedding_size),
                ('node', 'has', 'link'): GINLayer(node_embedding_size, node_embedding_size)}, aggr='sum'))

        # Readout Layer
        act = eval(act)
        for i in range(len(mlp_layers)):
            if mlp_bn:
                if i == 0:
                    self.readout.append(torch.nn.Sequential(torch.nn.Linear(node_embedding_size + self.concat_size + self.global_feats_size, mlp_layers[i]),
                                                            torch.nn.BatchNorm1d(num_features=mlp_layers[i]),
                                                            act
                                                            ))
                else:
                    self.readout.append(torch.nn.Sequential(torch.nn.Linear(mlp_layers[i - 1], mlp_layers[i]),
                                                            torch.nn.BatchNorm1d(num_features=mlp_layers[i]),
                                                            act
                                                            ))

            else:
                if i == 0:
                    self.readout.append(torch.nn.Sequential(torch.nn.Linear(node_embedding_size + self.concat_size + self.global_feats_size, mlp_layers[i]),
                                                            act
                                                            ))

                else:
                    self.readout.append(torch.nn.Sequential(torch.nn.Linear(mlp_layers[i - 1], mlp_layers[i]),
                                                            act
                                                            ))

        if mlp_head_act is None:
            self.readout.append(torch.nn.Sequential(torch.nn.Linear(mlp_layers[-1], 1)))
        else:
            self.readout.append(torch.nn.Sequential(torch.nn.Linear(mlp_layers[-1], 1),
                                                    eval(mlp_head_act)))

    def forward(self, x_dict, edge_index_dict, path_batch):
        if not self.divided_features:
            x_dict['path'] = torch.cat([x_dict['path'][:,0:3], x_dict['path'][:,6].reshape(-1,1)], axis=1)
            x_dict['link'] = torch.cat([x_dict['link'][:,0:3], x_dict['link'][:,4:7]], axis=1)
            if not self.bl_features:
                x_dict['path'] = x_dict['path'][:,0:3]
                x_dict['link'] = x_dict['link'][:,0:3]
        else:
            if not self.bl_features:
                x_dict['path'] = x_dict['path'][:,0:6]
                x_dict['link'] = x_dict['link'][:,0:3]


        origin_input = x_dict.copy()

        if self.global_feats:
            mean_global_feats = torch_geometric.nn.global_mean_pool(origin_input["path"], path_batch)
            mean_global_max = torch_geometric.nn.global_max_pool(origin_input["path"], path_batch)

            mean_global_feats = torch.gather(mean_global_feats, 0, path_batch.unsqueeze(1).repeat(1, mean_global_feats.shape[1]))
            mean_global_max = torch.gather(mean_global_max, 0, path_batch.unsqueeze(1).repeat(1, mean_global_max.shape[1]))

        # message passing
        for i in range(self.num_layers):
            x_dict = self.convs[i](x_dict, edge_index_dict)

            for k, _ in x_dict.items():
                x_dict[k] = torch.nn.functional.dropout(x_dict[k], p=self.dropout, training=self.training)

        # readout
        if self.concat_path:
            if self.global_feats:
                x = torch.cat((x_dict['path'], origin_input['path'], mean_global_feats, mean_global_max), 1)
            else:
                x = torch.cat((x_dict['path'], origin_input['path']), 1)
        else:
            if self.global_feats:
                x = torch.cat((x_dict['path'], mean_global_feats, mean_global_max), 1)
            else:
                x = x_dict['path']

        for i in range(len(self.mlp_layers) + 1):
            x = self.readout[i](x)

        return x



class HetroGAT(torch.nn.Module):
    def __init__(self, input_channels: dict, node_embedding_size: int, message_passing_layers: int, dropout: float, heads: int, concat_path: bool,bl_features: bool, divided_features: bool, global_feats: bool,
                 mlp_layers: list, act, mlp_head_act, mlp_bn: bool):
        super().__init__()
        self.num_layers = message_passing_layers
        self.dropout = dropout
        self.concat_path = concat_path
        self.mlp_layers = mlp_layers
        self.global_feats = global_feats
        self.heads = heads
        self.bl_features = bl_features
        self.divided_features = divided_features

        if global_feats:
            self.global_feats_size = 8
        else:
            self.global_feats_size = 0

        if concat_path:
            if self.divided_features and self.bl_features:
                self.concat_size = input_channels['path']
            elif self.divided_features and not self.bl_features:
                self.concat_size = input_channels['path'] - 1
            elif not self.divided_features and self.bl_features:
                self.concat_size = input_channels['path'] - 3
            else:
                self.concat_size = input_channels['path'] - 4

        else:
            self.concat_size = 0

        self.convs = torch.nn.ModuleList()
        self.readout = torch.nn.ModuleList()

        # Message Passing
        # first conv layer
        self.convs.append(HeteroConv({
            ('path', 'uses', 'link'): GATConv((-1, -1), node_embedding_size, heads=heads, concat=True),
            ('link', 'includes', 'path'): GATConv((-1, -1), node_embedding_size, heads=heads, concat=True),
            ('link', 'connects', 'node'): GATConv((-1, -1), node_embedding_size, heads=heads, concat=True),
            ('node', 'has', 'link'): GATConv((-1, -1), node_embedding_size, heads=heads, concat=True)}, aggr='sum'))

        # remaining conv layers
        for _ in range(self.num_layers - 1):
            self.convs.append(HeteroConv({
                ('path', 'uses', 'link'): GATConv(node_embedding_size, node_embedding_size),
                ('link', 'includes', 'path'): GATConv(node_embedding_size, node_embedding_size),
                ('link', 'connects', 'node'): GATConv(node_embedding_size, node_embedding_size),
                ('node', 'has', 'link'): GATConv(node_embedding_size, node_embedding_size)}, aggr='sum'))

        # Readout Layer
        act = eval(act)
        for i in range(len(mlp_layers)):
            if mlp_bn:
                if i == 0:
                    self.readout.append(torch.nn.Sequential(torch.nn.Linear(node_embedding_size * heads + self.concat_size + self.global_feats_size, mlp_layers[i]),
                                                            torch.nn.BatchNorm1d(num_features=mlp_layers[i]),
                                                            act
                                                            ))
                else:
                    self.readout.append(torch.nn.Sequential(torch.nn.Linear(mlp_layers[i - 1], mlp_layers[i]),
                                                            torch.nn.BatchNorm1d(num_features=mlp_layers[i]),
                                                            act
                                                            ))

            else:
                if i == 0:
                    self.readout.append(torch.nn.Sequential(torch.nn.Linear(node_embedding_size * heads + self.concat_size + self.global_feats_size, mlp_layers[i]),
                                                            act
                                                            ))

                else:
                    self.readout.append(torch.nn.Sequential(torch.nn.Linear(mlp_layers[i - 1], mlp_layers[i]),
                                                            act
                                                            ))

        if mlp_head_act is None:
            self.readout.append(torch.nn.Sequential(torch.nn.Linear(mlp_layers[-1], 1)))
        else:
            self.readout.append(torch.nn.Sequential(torch.nn.Linear(mlp_layers[-1], 1),
                                                    eval(mlp_head_act)))

    def forward(self, x_dict, edge_index_dict, path_batch):
        if not self.divided_features:
            x_dict['path'] = torch.cat([x_dict['path'][:,0:3], x_dict['path'][:,6].reshape(-1,1)], axis=1)
            x_dict['link'] = torch.cat([x_dict['link'][:,0:3], x_dict['link'][:,4:7]], axis=1)
            if not self.bl_features:
                x_dict['path'] = x_dict['path'][:,0:3]
                x_dict['link'] = x_dict['link'][:,0:3]
        else:
            if not self.bl_features:
                x_dict['path'] = x_dict['path'][:,0:6]
                x_dict['link'] = x_dict['link'][:,0:3]
                

        origin_input = x_dict.copy()

        if self.global_feats:
            mean_global_feats = torch_geometric.nn.global_mean_pool(origin_input["path"], path_batch)
            mean_global_max = torch_geometric.nn.global_max_pool(origin_input["path"], path_batch)

            mean_global_feats = torch.gather(mean_global_feats, 0, path_batch.unsqueeze(1).repeat(1, mean_global_feats.shape[1]))
            mean_global_max = torch.gather(mean_global_max, 0, path_batch.unsqueeze(1).repeat(1, mean_global_max.shape[1]))

        # message passing
        for i in range(self.num_layers):
            x_dict = self.convs[i](x_dict, edge_index_dict)

            for k, _ in x_dict.items():
                x_dict[k] = torch.nn.functional.dropout(x_dict[k], p=self.dropout, training=self.training)

        # readout
        if self.concat_path:
            if self.global_feats:
                x = torch.cat((x_dict['path'], origin_input['path'], mean_global_feats, mean_global_max), 1)
            else:
                x = torch.cat((x_dict['path'], origin_input['path']), 1)
        else:
            if self.global_feats:
                x = torch.cat((x_dict['path'], mean_global_feats, mean_global_max), 1)
            else:
                x = x_dict['path']

        for i in range(len(self.mlp_layers) + 1):
            x = self.readout[i](x)

        return x

