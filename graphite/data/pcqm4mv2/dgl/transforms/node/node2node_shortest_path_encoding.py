from typing import List

import dgl
import torch
import torch.nn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import numpy

from graphite.data.pcqm4mv2.transforms.base import BaseGraphitePCQM4MTransform
import pyximport
pyximport.install(setup_args={"include_dirs": numpy.get_include()})
from graphite.contrib.graphormer.algos import gen_edge_input, floyd_warshall


class EncodeNode2NodeShortestPathFeatureTrajectory(BaseGraphitePCQM4MTransform):
    def __init__(
            self,
            max_length_considered: int = 256
    ):
        """
        __WARNING__: This transform must be applied BEFORE adding additional node/edge. Otherwise,
        the results could very well be incorrect.
        :param max_length_considered:
        """
        super(EncodeNode2NodeShortestPathFeatureTrajectory, self).__init__()
        self.max_length_considered = max_length_considered
        self.unreachable = max_length_considered + 1
        self.task_distance = max_length_considered + 2
        self.no_distance = max_length_considered + 3  # through collate

    def forward(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        device = g.device

        edge_features = torch.zeros((g.num_nodes(), g.num_nodes(), 3), dtype=torch.long)
        e0, e1 = g.edges()
        edge_features[e0, e1, :] = g.edata['feat']
        M, path = floyd_warshall(g.adjacency_matrix().to_dense().long().data.cpu().numpy())

        g.ndata['shortest_path_feature_trajectory'] = torch.from_numpy(gen_edge_input(numpy.amax(M), path, edge_features.data.cpu().numpy().astype('int'))).to(device)
        return g
