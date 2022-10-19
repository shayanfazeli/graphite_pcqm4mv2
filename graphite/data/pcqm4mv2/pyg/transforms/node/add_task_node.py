from typing import List

from torch_geometric.data import Data
import torch
import torch.nn
import graphite.data.utilities.pcqm4mv2_meta as PCQM4MV2_METADATA
from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform


class AddTaskNode(BasePygGraphitePCQM4MTransform):
    def __init__(
            self,
            feat=PCQM4MV2_METADATA.task_node_feat
    ):
        """constructor"""
        super(AddTaskNode, self).__init__()
        self.feat = feat

    def forward(self, g: Data) -> Data:
        g['x'] = torch.cat((g.x, self.feat * torch.ones((1, g.x.shape[1]), dtype=g.x.dtype).to(g.x.device)), dim=0)
        g['node_type'] = torch.cat((g.node_type, self.feat * torch.ones((1, g.node_type.shape[1]), dtype=g.node_type.dtype).to(g.node_type.device)), dim=0)
        return g
