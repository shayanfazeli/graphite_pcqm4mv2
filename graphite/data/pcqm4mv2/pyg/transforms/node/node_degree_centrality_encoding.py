from torch_geometric.data import Data
from torch_geometric.utils import degree
import torch
import torch.nn
from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform


class EncodeNodeDegreeCentrality(BasePygGraphitePCQM4MTransform):
    """
    Degree centrality encoding
    """
    def __init__(
            self,
    ):
        """constructor"""
        super(EncodeNodeDegreeCentrality, self).__init__()

    def forward(self, g: Data) -> Data:
        """
        The transform forward

        Parameters
        ----------
        g: `Data`, required
            The PyG graph with the `x` feature PRIOR to adding task node (if any).

        Returns
        ----------
        `Data`: the `node_type` tensor is added to the graph and it is returned.
        """
        e0, e1 = tuple([e.squeeze() for e in torch.split(g.edge_index,  1, 0)])
        g['node_degree_centrality'] = degree(e0, num_nodes=g.num_nodes)
        return g
