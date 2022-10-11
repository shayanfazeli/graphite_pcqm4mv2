import torch
import torch.nn
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph

from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform


class LineGraphTransform(BasePygGraphitePCQM4MTransform):
    def __init__(
            self,
            bring_in_adjacent_nodes: bool
    ):
        """
        The core line graph transform with the option of bringing in the node features in the edge
        for a more expressive node features in the corresponding line graph.
        :param bring_in_adjacent_nodes:
        """
        super(LineGraphTransform, self).__init__()
        self.bring_in_adjacent_nodes = bring_in_adjacent_nodes
        self.line_graph_transform = LineGraph()

    def forward(self, g: Data) -> Data:
        """
        Please note that all the other tensors
        besides the `x`, `edge_index`, and `edge_attr`
        will be removed in the output of this transform.

        Parameters
        ----------
        g: `Data`, required
            The input graph

        Returns
        ----------
        `Data`:
            The modified transformer that now has the corresponding 3d edges.
        """
        x, edge_index, edge_attr = g.x, g.edge_index, g.edge_attr

        if self.bring_in_adjacent_nodes:
            edge_attr = torch.cat((edge_attr, x[edge_index, :].permute(1, 0, 2).flatten(1)), dim=1)

        return self.line_graph_transform(Data(
            x=x,
            edge_attr=edge_attr,
            edge_index=edge_index
        ))
