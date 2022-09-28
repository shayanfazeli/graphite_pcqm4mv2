from typing import List

from torch_geometric.data import Data
import torch
import torch.nn

from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform


class AddTaskNode(BasePygGraphitePCQM4MTransform):
    def __init__(
            self,
            feat=333
    ):
        """constructor"""
        super(AddTaskNode, self).__init__()
        self.feat = feat

    def forward(self, g: Data) -> Data:
        g['x'] = torch.cat((g.x, self.feat * torch.ones((1, g.x.shape[1]), dtype=g.x.dtype).to(g.x.device)), dim=0)
        return g
