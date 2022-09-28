from typing import List

from torch_geometric.data import Data
import torch
import torch.nn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import numpy

from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform


def compute_distance_matrix(g:  Data):
    row, col = tuple([e.squeeze() for e in torch.split(g.edge_index,  1, 0)])
    device = row.device
    row = row.data.cpu().numpy()
    col = col.data.cpu().numpy()
    weight = numpy.ones_like(row)

    num_nodes = g.num_nodes

    graph = csr_matrix((weight, (row, col)), shape=(num_nodes, num_nodes))
    dist_matrix, _ = shortest_path(
        csgraph=graph, directed=False, return_predecessors=True
    )

    return torch.from_numpy(dist_matrix).to(device)


class EncodeNode2NodeShortestPathLengthType(BasePygGraphitePCQM4MTransform):
    def __init__(
            self,
            max_length_considered: int = 5
    ):
        """constructor"""
        super(EncodeNode2NodeShortestPathLengthType, self).__init__()
        self.max_length_considered = max_length_considered
        self.unreachable = max_length_considered + 1
        self.task_distance = max_length_considered + 2
        self.no_distance = max_length_considered + 3  # through collate

    def forward(self, g: Data, task_node_added: bool = True) -> Data:
        """
        Parameters
        ----------
        g: `Data`, required
            The PyG graph with the `x` feature PRIOR to adding task node (if any).

        Returns
        ----------
        `Data`: the task node is added as the last node of the graph.
        """
        device = g.x.device

        if 'distance' in g:
            distance_matrix = g['distance']  # n, n
        else:
            distance_matrix = compute_distance_matrix(g)
        distance_matrix[distance_matrix == numpy.inf] = -1  # n, n
        distance_matrix = distance_matrix.clip(max=self.max_length_considered)
        distance_matrix[distance_matrix == -1] = self.unreachable

        # - filling with encoded edge types (scalars)
        connection_types = distance_matrix.to(device).long()

        # - task edge in the end
        if task_node_added:
            # - we then know the last node is the task node
            connection_types[-1, :-1] = self.task_distance
            connection_types[:-1, -1] = self.task_distance

        # - setting the new feature
        g['node2node_shortest_path_length_type'] = connection_types

        return g
