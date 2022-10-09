from typing import List
from torch_geometric.utils import coalesce
from torch_geometric.data import Data
import torch
from torch_cluster import radius_graph

from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform


class RadiusGraphEdges(BasePygGraphitePCQM4MTransform):
    """
    __Remark__: In case of training with a transformer-like architecture, it is common practice
    to add a virtual (task) node. Please make sure to apply this transform early if you are considering
    a transform pipeline, and before such modifications have altered the graph.

    Parameters
    ----------
    save_key: `str`, optional (default='3d_radius_edge_index')
        As in the other transforms, the output is going to be of
        type `torch_geometric.data.Data`. Thus, the `save_key` argument
        determines where in the graph the outputs of this transform are to be stored.

    positions_3d_key: `str`, optional (default='positions_3d')
        The key that is of `dim=(num_nodes, 3)` which contains
        the 3d positions.

    cutoff: `float`, optional (default=8.0)
        The cutoff value for determining the edges.

    merge_with_existing_edges: `bool`, optional (default=False)
        If this parameter is true, incase a key containing a tensor
        of `dim=(2, m1)` corresponding to edges already exist in the `save_key` place,
        it will compute the resulting edges from the radius graph and add them to the
        already existing edges.
        The use of this argument, for example, is to combine the edges from the radius graph with those coming
        from the `mol.GetBonds()` method.
    """
    def __init__(
            self,
            save_key: str = '3d_radius_edge_index',
            positions_3d_key: str = 'positions_3d',
            cutoff: float = 8.0,
            merge_with_existing_edges: bool = False
    ):
        """constructor"""
        super(RadiusGraphEdges, self).__init__()
        self.save_key = save_key
        self.positions_3d_key = positions_3d_key
        self.merge_with_existing_edges = merge_with_existing_edges
        self.cutoff = cutoff

    def forward(self, g: Data) -> Data:
        """
        Parameters
        ----------
        g: `Data`, required
            The input graph

        Returns
        ----------
        `Data`:
            The modified transformer that now has the corresponding 3d edges.
        """
        edges_3d = radius_graph(g[self.positions_3d_key], r=self.cutoff)
        if self.save_key in g and self.merge_with_existing_edges:
            edges_3d = coalesce(torch.cat((edges_3d, g[self.save_key]), dim=1))
        g[self.save_key] = edges_3d.detach()
        return g
