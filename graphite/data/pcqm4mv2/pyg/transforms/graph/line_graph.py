from typing import List
import torch
import torch.nn
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch_geometric.utils import coalesce, add_self_loops
from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform


class LineGraphTransform(BasePygGraphitePCQM4MTransform):
    def __init__(
            self,
            bring_in_adjacent_nodes: bool,
            keep_as_is: List[str] = ['fingerprint', 'molecule_descriptor', 'y']
    ):
        """
        The core line graph transform with the option of bringing in the node features in the edge
        for a more expressive node features in the corresponding line graph.
        :param bring_in_adjacent_nodes:
        """
        super(LineGraphTransform, self).__init__()
        self.bring_in_adjacent_nodes = bring_in_adjacent_nodes
        self.keep_as_is = keep_as_is
        self.line_graph_transform = LineGraph(force_directed=True)

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

        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=g.num_nodes, fill_value=-1)
        edge_index, edge_attr = coalesce(edge_index, edge_attr=edge_attr)

        if self.bring_in_adjacent_nodes:
            edge_attr = torch.cat((edge_attr, x[edge_index, :].permute(1, 0, 2).flatten(1)), dim=1)

        output = self.line_graph_transform(Data(
            x=x,
            edge_attr=edge_attr,
            edge_index=edge_index
        ))

        for c in self.keep_as_is:
            if c in g:
                output[c] = g[c]

        return output
