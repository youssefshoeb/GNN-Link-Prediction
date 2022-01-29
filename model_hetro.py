import copy
import torch
import torch_geometric
from typing import Optional, Callable, List, Any, Union
from torch_sparse import SparseTensor, matmul
from torch.functional import Tensor
from torch_geometric.nn import Linear
from torch_geometric.nn.conv.hetero_conv import HeteroConv
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.utils import to_dense_adj


torch.manual_seed(24)


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
    """A single GIN layer consits of the GINConv layer followed by batch normalization
    and prelu activation.
    """
    def __init__(self, in_channels: int, out_channels: int, concat: bool = False) -> None:
        super().__init__()

        # Mlp that maps node features x of shape [-1, in_channels] to shape [-1, out_channels] )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(out_channels, out_channels)
        )

        self.conv = GINConv(self.mlp, eps=0, train_eps=True, concat=concat)
        # self.act = torch.nn.PReLU()
        # self.bn = torch_geometric.nn.norm.BatchNorm(out_channels)

    def forward(self, x: Tensor, edge_index: Adj):
        x = self.conv(x, edge_index)
        # x = self.bn(x)
        # x = self.act(x)

        return x


class Hetro_GIN(torch.nn.Module):
    def __init__(self, input_channels: dict, embedding_size: int, num_layers: int = 2,
                 dropout: float = 0.0,
                 act: Optional[Callable] = torch.nn.ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 post_hidden_layer_size: int = 16, post_num_layers: int = 2):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act
        self.norm = norm
        self.jk_mode = jk

        self.convs = torch.nn.ModuleList()

        # first conv layer
        self.convs.append(HeteroConv({
            ('path', 'uses', 'link'): GINLayer(input_channels['path'] + input_channels['link'], embedding_size, concat=True),
            ('link', 'includes', 'path'): GINLayer(input_channels['link'] + input_channels['path'], embedding_size, concat=True),
            ('link', 'connects', 'node'): GINLayer(input_channels['link'] + input_channels['node'], embedding_size, concat=True),
            ('node', 'has', 'link'): GINLayer(input_channels['node'] + input_channels['link'], embedding_size, concat=True)}, aggr='sum'))

        # remaining conv layers
        for i in range(num_layers - 1):
            self.convs.append(HeteroConv({
                ('path', 'uses', 'link'): GINLayer(embedding_size, embedding_size),
                ('link', 'includes', 'path'): GINLayer(embedding_size, embedding_size),
                ('link', 'connects', 'node'): GINLayer(embedding_size, embedding_size),
                ('node', 'has', 'link'): GINLayer(embedding_size, embedding_size)}, aggr='sum'))

        # batch normalization
        if norm is not None:
            self.norms = torch.nn.ModuleList()
            for _ in range(num_layers):
                self.norms.append(copy.deepcopy(norm))

        # jumping knowledge
        if self.jk_mode != 'last':
            self.jk = JumpingKnowledge(jk, embedding_size, num_layers)
        if self.jk_mode == 'cat':
            self.lin = Linear(num_layers * embedding_size, embedding_size)

        # post_processing
        self.post_num_layers = post_num_layers

        self.post_layers = torch.nn.ModuleList()

        self.post_layers.append(torch.nn.Sequential(torch.nn.Linear(embedding_size, post_hidden_layer_size),
                                torch.nn.BatchNorm1d(post_hidden_layer_size),
                                torch.nn.PReLU()))

        for _ in range(post_num_layers - 2):
            self.post_layers.append(torch.nn.Sequential(torch.nn.Linear(post_hidden_layer_size, post_hidden_layer_size),
                                    torch.nn.BatchNorm1d(post_hidden_layer_size),
                                    torch.nn.PReLU()))

        self.post_layers.append(torch.nn.Sequential(torch.nn.Linear(post_hidden_layer_size, 1), torch.nn.ReLU()))

    def forward(self, x_dict, edge_index_dict):
        # TODO Double check this
        xs_dict: List[Tensor] = {k: [] for k, _ in x_dict.items()}
        for i in range(self.num_layers):
            x_dict = self.convs[i](x_dict, edge_index_dict)

            if self.norms is not None:
                for k, _ in x_dict.items():
                    x_dict[k] = self.norms[i](x_dict[k])

            if self.act is not None:
                for k, _ in x_dict.items():
                    x_dict[k] = self.act(x_dict[k])

            for k, _ in x_dict.items():
                x_dict[k] = torch.nn.functional.dropout(x_dict[k], p=self.dropout, training=self.training)

            if hasattr(self, 'jk'):
                for k, _ in x_dict.items():
                    xs_dict[k].append(x_dict[k])

        for k, _ in x_dict.items():
            x_dict[k] = self.jk(xs_dict[k]) if hasattr(self, 'jk') else x_dict[k]
            x_dict[k] = self.lin(x_dict[k]) if hasattr(self, 'lin') else x_dict[k]

        # post_processing
        x = x_dict['link']
        for i in range(self.post_num_layers):
            x = self.post_layers[i](x)

        # Get path delays
        # TODO: check if gradients are propagated correctly in this way (maybe find a differnt way using scatter ?)
        # delays = torch.zeros(x_dict['path'].shape[0], dtype=torch.float)

        # for index, path in enumerate(edge_index_dict[('path', 'uses', 'link')][0]):
        #     link = edge_index_dict[('path', 'uses', 'link')][1][index]
        #     delays.index_add_(0, path.cpu(), x[link][0].cpu())

        return x
