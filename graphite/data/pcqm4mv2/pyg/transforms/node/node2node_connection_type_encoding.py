from typing import List

from torch_geometric.data import Data
import torch
import torch.nn

from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform


class EncodeNode2NodeConnectionType(BasePygGraphitePCQM4MTransform):
    def __init__(
            self,
            num_edge_types: int = 27
    ):
        """constructor"""
        super(EncodeNode2NodeConnectionType, self).__init__()
        self.num_edge_types = num_edge_types
        self.task_edge_type = self.num_edge_types + 1
        self.self_edge_type = self.num_edge_types + 2
        self.no_edge_type = self.num_edge_types + 3  # this should be done for collation

    def forward(self, g: Data, task_node_added: bool = True) -> Data:
        """
        The transform forward

        Parameters
        ----------
        g: `Data`, required
            The PyG graph of a molecule

        Returns
        ----------
        `Data`: the `node_type` tensor is added to the graph and it is returned.
        """
        assert g.edge_attr.ndim == 1, "you have  to have a long 1-dim edge attribute (type)"
        device = g.x.device
        edge_types = g.edge_attr  # m
        e0, e1 = tuple([e.squeeze() for e in torch.split(g.edge_index,  1, 0)])
        n = g.num_nodes

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
        g['node2node_connection_type'] = connection_types

        return g
