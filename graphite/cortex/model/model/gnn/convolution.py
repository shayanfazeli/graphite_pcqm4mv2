from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.utils import degree

from graphite.cortex.model.model.gnn.model import LineGraphNodeRepresentation


class GINConv(MessagePassing):
    """
    GIN Convolution (a message passing scheme)

    Parameters
    ----------
    model_dim: `int`, required
        The core dimensions
    """
    def __init__(
            self,
            model_dim: int,
            pos_features: int = None
    ):
        """constructor"""
        super(GINConv, self).__init__(aggr="add")
        # - core mlp
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.BatchNorm1d(model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, model_dim)
        )
        # - gin eps
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        if pos_features is not None:
            self.pos_encoder = torch.nn.Sequential(
                torch.nn.Linear(pos_features, model_dim),
                torch.nn.LayerNorm(model_dim),
            )
        self.pos_features = pos_features

        # - bond encoder for pcqm4mv2 edge features
        self.bond_encoder = BondEncoder(emb_dim=model_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        # - encoding the edges
        edge_embedding = self.bond_encoder(edge_attr[:, :3].long())
        if self.pos_features is not None:
            edge_embedding = edge_embedding + self.pos_encoder(edge_attr[:, 3:])

        # - propagation and saving the outputs (the new node representations)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out: torch.Tensor):
        return aggr_out


class GINConvForLineGraph(MessagePassing):
    """
    GIN Convolution (a message passing scheme)

    Parameters
    ----------
    model_dim: `int`, required
        The core dimensions
    """
    def __init__(
            self,
            model_dim: int
    ):
        """constructor"""
        super(GINConvForLineGraph, self).__init__(aggr="add")
        # - core mlp
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(model_dim, model_dim),
            torch.nn.BatchNorm1d(model_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_dim, model_dim)
        )
        # - gin eps
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        # - propagation and saving the outputs (the new node representations)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x))
        return out

    def message(self, x_j: torch.Tensor):
        return F.relu(x_j)

    def update(self, aggr_out: torch.Tensor):
        return aggr_out


class GCNConv(MessagePassing):
    """
    Graph convolutional neural network

    Parameters
    ----------
    model_dim: `int`, required
        The core dimensions
    """
    def __init__(self, model_dim: int, pos_features: int = None):
        super(GCNConv, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(model_dim, model_dim)
        self.root_emb = torch.nn.Embedding(1, model_dim)
        self.bond_encoder = BondEncoder(emb_dim=model_dim)
        if pos_features is not None:
            self.pos_encoder = torch.nn.Sequential(
                torch.nn.Linear(pos_features, model_dim),
                torch.nn.LayerNorm(model_dim),
            )
        self.pos_features = pos_features

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr[:, :3].long())
        if self.pos_features is not None:
            edge_embedding = edge_embedding + self.pos_encoder(edge_attr[:, 3:])

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor, norm: torch.Tensor):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out: torch.Tensor):
        return aggr_out


class GCNConvForLineGraph(MessagePassing):
    """
    Graph convolutional neural network

    Parameters
    ----------
    model_dim: `int`, required
        The core dimensions
    """
    def __init__(self, model_dim: int):
        super(GCNConvForLineGraph, self).__init__(aggr='add')
        self.linear = torch.nn.Linear(model_dim, model_dim)
        self.root_emb = torch.nn.Embedding(1, model_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        x = self.linear(x)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm) + F.relu(
            x + self.root_emb.weight) * 1. / deg.view(-1, 1)

    def message(self, x_j: torch.Tensor, norm: torch.Tensor):
        return norm.view(-1, 1) * F.relu(x_j)

    def update(self, aggr_out: torch.Tensor):
        return aggr_out


class GNN(torch.nn.Module):
    """
    Graph neural network: putting the graph convolutional network and gin module to use in a
    deep module.

    Parameters
    ----------
    num_layers: `int`, required
        The number of layers

    model_dim: `int`, required
        The core dimensions

    drop_ratio: `float`, optional (default=0.5)

    JK: `str`, optional (default="last")
        Obtaining the final node representations

    residual: `bool`, optional (default=False)
        Residual connections

    gnn_type: `str`, optional (default='gin')
        Core GNN module (`gcn` or `gin`)

    line_graph: `bool`, required
        If true, node embeddings must have been already done elsewhere. otherwise, the atom encoding
        of OGB will be used prior to the model application.
    """

    def __init__(
            self,
            num_layers: int,
            model_dim: int,
            drop_ratio: float = 0.5,
            JK: str = "last",
            residual: bool = False,
            gnn_type: str = 'gin',
            line_graph: bool = False,
            pos_features: int = None
    ):
        """constructor"""
        super(GNN, self).__init__()

        if line_graph:
            assert pos_features is None

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # - atom encoder
        self.line_graph = line_graph
        if not self.line_graph:
            self.atom_encoder = AtomEncoder(model_dim)
        else:
            self.atom_encoder = LineGraphNodeRepresentation(width=model_dim, width_head=1, width_scale=1)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layers):
            if not self.line_graph:
                if gnn_type == 'gin':
                    self.convs.append(GINConv(model_dim, pos_features))
                elif gnn_type == 'gcn':
                    self.convs.append(GCNConv(model_dim, pos_features))
                else:
                    ValueError('Undefined GNN type called {}'.format(gnn_type))
            else:
                if gnn_type == 'gin':
                    self.convs.append(GINConvForLineGraph(model_dim))
                elif gnn_type == 'gcn':
                    self.convs.append(GCNConvForLineGraph(model_dim))
                else:
                    ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(model_dim))

    def forward(self, batched_data: Batch):
        # - getting the core components
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        # - layer by layer, retrieving and saving the results
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            # - performing the core message passing and update
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            # - performing batchnorm
            h = self.batch_norms[layer](h)

            # - activation (except last layer) and dropout
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            # - resnet connection
            if self.residual:
                h += h_list[layer]

            # - storing it
            h_list.append(h)

        # - aggregation for final node representatioons
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]

        return node_representation


class GNNWithVirtualNode(torch.nn.Module):
    """

    Parameters
    ----------
    num_layers: `int`, required
        The number of layers

    model_dim: `int`, required
        The core dimensions

    drop_ratio: `float`, optional (default=0.5)

    JK: `str`, optional (default="last")
        Obtaining the final node representations

    residual: `bool`, optional (default=False)
        Residual connections

    gnn_type: `str`, optional (default='gin')
        Core GNN module (`gcn` or `gin`)

    line_graph: `bool`, required
        If true, node embeddings must have been already done elsewhere. otherwise, the atom encoding
        of OGB will be used prior to the model application.
    """

    def __init__(
            self,
            num_layers: int,
            model_dim: int,
            drop_ratio=0.5,
            JK: str = "last",
            residual: bool = False,
            gnn_type: str = 'gin',
            line_graph: bool = False,
            pos_features: int = None
    ):
        """constructor"""

        super(GNNWithVirtualNode, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.line_graph = line_graph
        if not self.line_graph:
            self.atom_encoder = AtomEncoder(model_dim)
        else:
            self.atom_encoder = LineGraphNodeRepresentation(width=model_dim, width_head=1, width_scale=1)

        # - setting the virtual node embedding
        self.virtualnode_embedding = torch.nn.Embedding(1, model_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # - layer convs
        self.convs = torch.nn.ModuleList()
        # - layer batch norms
        self.batch_norms = torch.nn.ModuleList()

        # - transform on the virtual node for each layer
        self.mlp_virtualnode_list = torch.nn.ModuleList()

        # - building layer modules
        for layer in range(num_layers):
            if not self.line_graph:
                if gnn_type == 'gin':
                    self.convs.append(GINConv(model_dim, pos_features=pos_features))
                elif gnn_type == 'gcn':
                    self.convs.append(GCNConv(model_dim, pos_features=pos_features))
                else:
                    ValueError('Undefined GNN type called {}'.format(gnn_type))
            else:
                if gnn_type == 'gin':
                    self.convs.append(GINConvForLineGraph(model_dim))
                elif gnn_type == 'gcn':
                    self.convs.append(GCNConvForLineGraph(model_dim))
                else:
                    ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(model_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(
                torch.nn.Sequential(
                    torch.nn.Linear(model_dim, model_dim),
                    torch.nn.BatchNorm1d(model_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(model_dim, model_dim),
                    torch.nn.BatchNorm1d(model_dim), torch.nn.ReLU()
                )
            )

    def forward(self, batched_data: Batch):
        # - getting core components
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        # - start from a randomly initialized vector, and broadcast it to all graphs
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))

        # - same as gnn, start layer by layer building the node representations
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            # - first, "broadcast" virtual node embedding to all nodes and sum it
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            # - message passing
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            # - non-linearity and dropout
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            # - residual
            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            # - updating virtual node embedding
            if layer < self.num_layers - 1:
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding
                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp), self.drop_ratio,
                        training=self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                                                      self.drop_ratio, training=self.training)

        # - obtaining node representations
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers + 1):
                node_representation += h_list[layer]

        return node_representation
