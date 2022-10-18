from typing import Dict, Any
import torch
from torch_geometric.data import Data
import pytorch_metric_learning.distances as dist_lib
from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform


class Position3DGaussianNoise(BasePygGraphitePCQM4MTransform):
    def __init__(
            self,
            scale: float
    ):
        """constructor"""
        super(Position3DGaussianNoise, self).__init__()
        self.scale = scale

    def forward(self, g: Data) -> Data:
        assert 'positions_3d' in g
        g.positions_3d = g.positions_3d + self.scale * torch.randn(g.positions_3d.size())
        return g
