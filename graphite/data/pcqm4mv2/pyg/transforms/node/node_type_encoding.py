from typing import List
from torch_geometric.data import Data
import torch
import torch.nn
from graphite.data.pcqm4mv2.pyg.transforms.base import BasePygGraphitePCQM4MTransform
from graphite.utilities.offset import add_feature_position_offset


class EncodeNodeType(BasePygGraphitePCQM4MTransform):
    """
    Encoding the node types of PCQM4M:

    In this version of the encoding, we follow OGB's protocol (which was also used in
    graphormer and grpe works) to embed the node features. That is,
    given that the highest number of unique values per node is 36, we choose the
    offset of 37 and create corresponding offsets to separate the
    embedding layout of each value.
    """
    def __init__(
            self,
            vocabulary_lengths: List[int] = [
                36, 3, 7, 7, 5, 4, 6, 2, 2
            ],
    ):
        """constructor"""
        super(EncodeNodeType, self).__init__()
        self.vocabulary_lengths = vocabulary_lengths
        self.offset = max(vocabulary_lengths) + 1
        self.num_node_types = self.offset * len(vocabulary_lengths) # 332
        self.task_node_type = self.num_node_types + 1  # 333

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
        # g['node_type'] = self.convert_to_id(g.x)
        g['node_type'] = add_feature_position_offset(g.x, offset=self.offset)
        return g
