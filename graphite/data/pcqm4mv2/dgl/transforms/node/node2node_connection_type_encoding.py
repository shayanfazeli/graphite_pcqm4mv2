from typing import List

import dgl
import torch
import torch.nn

from graphite.data.pcqm4mv2.transforms.base import BaseGraphitePCQM4MTransform


class EncodeNode2NodeConnectionType(BaseGraphitePCQM4MTransform):
    def __init__(
            self,
            num_edge_types: int = 27
    ):
        super(EncodeNode2NodeConnectionType, self).__init__()
        self.num_edge_types = num_edge_types
        self.task_edge_type = self.num_edge_types + 1
        self.self_edge_type = self.num_edge_types + 2
        self.no_edge_type = self.num_edge_types + 3  # this should be done for collation

    def forward(self, g: dgl.DGLGraph, task_node_added: bool = True) -> dgl.DGLGraph:
        assert 'edge_type' in g.edata, "the `edge_types` must be present as an edge feature in the graph."
        device = g.edata['edge_type'].device
        edge_types = g.edata['edge_type'].squeeze()  # m, 1
        e0, e1 = g.edges()
        n = g.num_nodes()

        # - filling with encoded edge types (scalars)
        connection_types = torch.zeros((n, n)).long().to(device) + self.no_edge_type
        connection_types[e0, e1] = edge_types

        # - self-edge
        connection_types = connection_types.fill_diagonal_(self.self_edge_type)

        # - task edge in the end
        if task_node_added:
            # - we then know the last node is the task node
            connection_types[-1, :-1] = self.task_edge_type
            connection_types[:-1, -1] = self.task_edge_type

        # - setting the new feature
        g.ndata['node2node_connection_type'] = connection_types

        return g
