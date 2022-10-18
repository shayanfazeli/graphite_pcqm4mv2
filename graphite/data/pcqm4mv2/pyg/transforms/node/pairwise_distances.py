from typing import Dict, Any
import torch
from torch_geometric.data import Data
import pytorch_metric_learning.distances as dist_lib
from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform


class PairwiseDistances(BasePygGraphitePCQM4MTransform):
    def __init__(
            self,
            distance_config: Dict[str, Any] = dict(
                type='LpDistance',
                args=dict(
                    p=2,
                    power=1,
                    normalize_embeddings=False
                )
            )
    ):
        """constructor"""
        super(PairwiseDistances, self).__init__()
        self.distance = getattr(dist_lib, distance_config['type'])(**distance_config['args'])

    def forward(self, g: Data) -> Data:
        assert 'positions_3d' in g
        g.pairwise_distances = self.distance(g.positions_3d)
        return g
