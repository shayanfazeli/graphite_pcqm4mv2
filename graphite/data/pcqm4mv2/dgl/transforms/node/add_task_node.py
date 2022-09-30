from typing import List

import dgl
import torch
import torch.nn

from graphite.data.pcqm4mv2.transforms.base import BaseGraphitePCQM4MTransform


class AddTaskNode(BaseGraphitePCQM4MTransform):
    def __init__(
            self,
            feat=333 * torch.ones((1,9))
    ):
        super(AddTaskNode, self).__init__()
        self.feat = feat

    def forward(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        g.add_nodes(1, {'feat': self.feat.long()})
        return g
