from typing import List
import math
from torch_geometric.data import Data
import torch
import torch.nn
from torch_scatter import scatter, scatter_min
from graphite.contrib.comenet.positions import *
from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform
from graphite.utilities.logging import get_logger

logger = get_logger(__name__)


class ConcatenateAtomPositionsToEdgeAttributes(BasePygGraphitePCQM4MTransform):
    """
    The 3D edge features of ComENet `https://arxiv.org/pdf/2206.08515.pdf`.

    # todo: modify for customizeable application on different edge indices (possibly multiple)
    """
    def __init__(
            self,
            edge_index_key: str = 'edge_index',
            include_atom_types: bool = True
    ):
        """constructor"""
        super(ConcatenateAtomPositionsToEdgeAttributes, self).__init__()
        self.edge_index_key = edge_index_key
        self.include_atom_types = include_atom_types

    def forward(self, g: Data) -> Data:
        """
        Parameters
        ----------
        g: `Data`, required


        Returns
        ----------
        """
        # - required material
        pos = g.positions_3d
        j, i = tuple([e.squeeze() for e in torch.split(g[self.edge_index_key], 1, 0)])

        if self.include_atom_types:
            g.edge_attr = torch.cat((g.edge_attr, pos[j], pos[i], g.x[j, 0].unsqueeze(-1), g.x[i, 0].unsqueeze(-1)), dim=1)
        else:
            g.edge_attr = torch.cat((g.edge_attr, pos[j], pos[i]), dim=1)

        return g
