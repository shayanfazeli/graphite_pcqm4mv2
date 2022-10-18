from typing import Dict, Any
import torch
import torch.nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from graphite.cortex.model.model.gnn.convolution import GNN, GNNWithVirtualNode
from graphite.cortex.model.model.gnn.model import CoAtGIN

node_encoder_classes = {
    'GNN': GNN,
    'GNNWithVirtualNode': GNNWithVirtualNode,
    'CoAtGIN': CoAtGIN
}


class CoAtGINGeneralPipeline(torch.nn.Module):
    def __init__(
            self,
            node_encoder_config: Dict[str, Any] = dict(
                type='CoAtGIN',
                args=dict(
                    num_layers=5,
                    model_dim=256,
                    conv_hop=2,
                    conv_kernel=2,
                    use_virt=True,
                    use_att=True
                )
            ),
            graph_pooling: str = "sum"
    ):
        super(CoAtGINGeneralPipeline, self).__init__()
        self.model_dim = node_encoder_config['args']['model_dim']
        self.graph_pooling = graph_pooling

        if node_encoder_config['args']['num_layers'] < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn_node = node_encoder_classes[node_encoder_config['type']](**node_encoder_config['args'])

        # - pooling module for graph-level representation
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(
                torch.nn.Linear(self.model_dim, self.model_dim),
                torch.nn.BatchNorm1d(self.model_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.model_dim, 1)
            ))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(self.model_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.norm = torch.nn.GroupNorm(1, self.model_dim, affine=False)

    def forward(self, batched_data, return_node_reps: bool = False):
        h_nodes = self.gnn_node(batched_data)  # - batched node counts
        h = self.pool(h_nodes, batched_data.batch)
        h = self.norm(h)

        if return_node_reps:
            return h, h_nodes
        else:
            return h
