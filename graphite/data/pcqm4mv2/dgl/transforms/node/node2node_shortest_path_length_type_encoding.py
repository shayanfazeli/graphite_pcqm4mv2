from typing import List

import dgl
import torch
import torch.nn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import numpy

from graphite.data.pcqm4mv2.transforms.base import BaseGraphitePCQM4MTransform


def compute_distance_matrix(g):
    row, col = g.edges()
    device = row.device
    row = row.data.cpu().numpy()
    col = col.data.cpu().numpy()
    weight = numpy.ones_like(row)

    num_nodes = g.num_nodes()

    graph = csr_matrix((weight, (row, col)), shape=(num_nodes, num_nodes))
    dist_matrix, _ = shortest_path(
        csgraph=graph, directed=False, return_predecessors=True
    )

    return torch.from_numpy(dist_matrix).to(device)


class EncodeNode2NodeShortestPathLengthType(BaseGraphitePCQM4MTransform):
    def __init__(
            self,
            max_length_considered: int = 256
    ):
        super(EncodeNode2NodeShortestPathLengthType, self).__init__()
        self.max_length_considered = max_length_considered
        self.unreachable = max_length_considered + 1
        self.task_distance = max_length_considered + 2
        self.no_distance = max_length_considered + 3  # through collate

    def forward(self, g: dgl.DGLGraph, task_node_added: bool = True) -> dgl.DGLGraph:
        device = g.device

        if 'distance' in g.ndata:
            distance_matrix = g.ndata['distance']  # n, n
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
        g.ndata['node2node_shortest_path_length_type'] = connection_types

        return g