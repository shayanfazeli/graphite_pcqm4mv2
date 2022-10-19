from typing import List

from torch_geometric.data import Data
import torch
import torch.nn
import ogb.utils.features
import graphite.data.utilities.pcqm4mv2_meta as PCQM4MV2_META
from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform


class EncodeEdgeType(BasePygGraphitePCQM4MTransform):
    def __init__(
            self,
            vocabulary_lengths: List[int] = PCQM4MV2_META.bond_feature_dims,
    ):
        """constructor"""
        super(EncodeEdgeType, self).__init__()
        self.vocabulary_lengths = vocabulary_lengths
        self.offset = PCQM4MV2_META.bonf_feature_offset

    def convert_to_id(self, edge_features: torch.Tensor):
        offset = torch.ones((1, edge_features.shape[1]), dtype=torch.long)
        offset[:, 1:] = self.offset
        offset = torch.cumprod(offset, dim=1)
        return (edge_features * offset).sum(dim=1).long()

    def forward(self, g: Data) -> Data:
        """
        Parameters
        ----------
        g: `Data`, required


        Returns
        ----------
        """
        g['edge_attr'] = self.convert_to_id(edge_features=g.edge_attr)
        return g
