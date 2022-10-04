from torch_geometric.data import Data
from  torch_geometric.utils import to_dense_adj
import torch
import torch.nn
import numpy
from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform
import pyximport
from graphite.utilities.logging import get_logger
from graphite.utilities.offset import add_feature_position_offset

pyximport.install(setup_args={"include_dirs": numpy.get_include()})
from graphite.contrib.graphormer.algos import gen_edge_input, floyd_warshall

logger = get_logger(__name__)


class EncodeNode2NodeShortestPathFeatureTrajectory(BasePygGraphitePCQM4MTransform):
    def __init__(
            self,
            max_length_considered: int,
            feature_position_offset: int = None
    ):
        """constructor"""
        """
        __WARNING__: This transform must be applied BEFORE adding additional node/edge. Otherwise,
        the results could very well be incorrect.
        :param max_length_considered:
        """
        super(EncodeNode2NodeShortestPathFeatureTrajectory, self).__init__()
        self.max_length_considered = max_length_considered
        self.feature_position_offset = feature_position_offset

    def forward(self, g: Data) -> Data:
        """
        The transform forward

        Parameters
        ----------
        g: `Data`, required
            The PyG graph of a molecule

        Returns
        ----------
        `Data`: the `node_type` tensor is added to the graph and it is returned.
        """
        device = g.x.device

        edge_attr = g.edge_attr.clone()
        if self.feature_position_offset is not None:
            edge_attr = add_feature_position_offset(edge_attr, offset=self.feature_position_offset)

        edge_features = torch.zeros((g.num_nodes, g.num_nodes, edge_attr.shape[1]), dtype=torch.long)
        e0, e1 = tuple([e.squeeze() for e in torch.split(g.edge_index,  1, 0)])
        edge_features[e0, e1, :] = edge_attr
        if g.edge_index.shape[1] == 0:
            # logger.warning(f"ENCOUNTERED A GRAPH WITH {g.num_nodes} NODES AND NO EDGES.")
            adj = numpy.zeros((g.num_nodes, g.num_nodes)).astype('int64')
        else:
            adj = to_dense_adj(g.edge_index).squeeze(0).long().data.cpu().numpy()
        M, path = floyd_warshall(adj)

        g['shortest_path_feature_trajectory'] = torch.from_numpy(gen_edge_input(numpy.amax(M), path, edge_features.data.cpu().numpy().astype('int'))).to(device)
        g['shortest_path_feature_trajectory'] = g['shortest_path_feature_trajectory'][:, :, :self.max_length_considered, :]
        return g
