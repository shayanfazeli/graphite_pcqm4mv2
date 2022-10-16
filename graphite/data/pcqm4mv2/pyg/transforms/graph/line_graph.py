from typing import List
import torch
import torch.nn
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph
from torch_geometric.utils import coalesce, add_self_loops, degree
from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform


class LineGraphTransform(BasePygGraphitePCQM4MTransform):
    """
    Core transform for representing the PCQM4Mv2 graphs (with or without 3d information)
    as line graphs.

    __Remark__: Please note that due to the edge attribute of `[-1, -1, -1]` for
    self-edges (please refer to `self_loop_strategy` for more details), your
    edge encoder must support this in addition to the usual bond features of
    the dataset (examples would be padded embedding).

    Parameters
    ----------
    bring_in_adjacent_nodes: `bool`, required
        If `True`, each edge attribute gets concatenated with the feature vector
        of the two atoms it is connecting together.

    keep_as_is: `List[str]`, optional (default = ['fingerprint', 'molecule_descriptor', 'y'])
        The columns passed to this will be checked in the original graph (which is of `Data` type),
        and if the corresponding data is present it will be passed to the
        corresponding linegraph with no change.

    self_loop_strategy: `str`, optional (default = 'isolated_nodes_only')
        The self loop strategy indicates where to draw an edge with
        the attribute `[-1, -1, -1]` as a self-loop in the graph prior
        to conversion to line graph. This is to ensure that node information for
        isolated nodes don't get lost.

        * `isolated_nodes_only`: In each graph, if there are nodes with no edge (in or out), those will be
        served with a self-loop.
        * `all`: In all graphs, automatically a self-loop will be added to all nodes.
        * `no_edge_graphs`: The self-loop will be added to all nodes of a graph if there is no edge present.
        * `none`: no additional self-loop would be enforced.
    """
    def __init__(
            self,
            bring_in_adjacent_nodes: bool,
            keep_as_is: List[str] = ['fingerprint', 'molecule_descriptor', 'y'],
            self_loop_strategy: str = 'isolated_nodes_only'  # all,  none, no_edge_graphs
    ):
        """
        constructor
        """
        super(LineGraphTransform, self).__init__()
        self.bring_in_adjacent_nodes = bring_in_adjacent_nodes
        self.keep_as_is = keep_as_is
        self.line_graph_transform = LineGraph(force_directed=True)
        self.self_loop_strategy = self_loop_strategy

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

        if self.self_loop_strategy == 'all' or (
                self.self_loop_strategy == 'no_edge_graphs' and edge_index.shape[1] == 0):
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=g.num_nodes,
                                                   fill_value=-1)
            edge_index, edge_attr = coalesce(edge_index, edge_attr=edge_attr)

        elif self.self_loop_strategy == 'isolated_nodes_only':
            out_degree = degree(edge_index[0], num_nodes=x.shape[0])
            in_degree = degree(edge_index[1], num_nodes=x.shape[0])
            full_degree = in_degree + out_degree
            isolated_node_indices = (full_degree == 0).nonzero().view(-1)
            num_isolated_nodes = isolated_node_indices.shape[0]
            if num_isolated_nodes > 0:
                edge_index = torch.cat((edge_index, isolated_node_indices.unsqueeze(0).repeat(2, 1)), dim=1)
                edge_attr = torch.cat((edge_attr, torch.ones((num_isolated_nodes, edge_attr.shape[1]))), dim=0)
                edge_index, edge_attr = coalesce(edge_index, edge_attr=edge_attr)
        elif self.self_loop_strategy == 'none':
            pass
        else:
            raise Exception(f"unknown self loop strategy: {self.self_loop_strategy}")

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
