from typing import List

import dgl
import torch
import torch.nn

from graphite.data.pcqm4mv2.transforms.base import BaseGraphitePCQM4MTransform


class EncodeEdgeType(BaseGraphitePCQM4MTransform):
    def __init__(
            self,
            vocabulary_lengths: List[int] = [
                4, 3, 2
            ],
    ):
        super(EncodeEdgeType, self).__init__()
        self.vocabulary_lengths = vocabulary_lengths
        self.offset = 4
        self.num_edge_types = self.offset * len(vocabulary_lengths)

    def convert_to_id(self, edge_features: torch.Tensor):
        offset = torch.ones((1, edge_features.shape[1]), dtype=torch.long)
        offset[:, 1:] = self.offset
        offset = torch.cumprod(offset, dim=1)
        return (edge_features * offset).sum(dim=1).long()

    def forward(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        g.edata['edge_type'] = self.convert_to_id(edge_features=g.edata['feat'])
        return g
