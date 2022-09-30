from typing import List

import dgl
import torch
import torch.nn
import numpy

from graphite.data.pcqm4mv2.transforms.base import BaseGraphitePCQM4MTransform


class EncodeNodeType(BaseGraphitePCQM4MTransform):
    def __init__(
            self,
            vocabulary_lengths: List[int] = [
                36, 3, 7, 7, 5, 4, 6, 2, 2
            ],
    ):
        super(EncodeNodeType, self).__init__()
        self.vocabulary_lengths = vocabulary_lengths
        self.offset = max(vocabulary_lengths) + 1
        self.num_node_types = self.offset * len(vocabulary_lengths) # 332
        self.task_node_type = self.num_node_types + 1  # 333

    def convert_to_id(self, node_features: torch.Tensor):
        offset = self.offset
        feature_offset = 1 + torch.arange(
            0, len(self.vocabulary_lengths) * offset, offset, dtype=torch.long
        )
        node_features = node_features + feature_offset
        return node_features.long()

    def forward(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        g.ndata['node_type'] = self.convert_to_id(g.ndata['feat'])
        return g
