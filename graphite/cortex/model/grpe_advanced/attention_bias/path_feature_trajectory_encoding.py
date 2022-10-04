from graphite.cortex.model.grpe.attention_bias.base import AttentionBiasBase
import torch
import torch.nn


class PathTrajectoryEncodingAttentionBias(AttentionBiasBase):
    """
    This module's output corresponds to the path encoding of Graphormer.

    __Remark__: It is user's responsibility to decide whether a single embedding
    codebook is to be shared between each edge feature position (3 positions in
    PCQM4Mv2 corresponding to bond features), or whether separate embedding
    per feature position is needed. Note that once this module is initialized,
    the `self.codebook` will be used on all edge features, thus, if you intend
    to use separate embedding per feature index, make sure to use the dataset
    transform as follows:

    ```python
    EncodeNode2NodeShortestPathFeatureTrajectory(
        max_length_considered=4,
        feature_position_offset=4 # <-
    )
    ```

    This line ensures that, while edge features remain intact (and you can use them as intended by
    any transform that follows after this), the copy that is used in creating the path trajectories
    (which for every item is of dim `num_nodes, num_nodes, max_path_length, 3`), has gone through
    the proper offset which has made it ready to be embedded.

    Parameters
    ----------
    num_heads: `int`, required
        The number of attention heads for constructing the biases

    code_dim: `int`, required
        The representation dimension of the internal codebook for edge feature embeddings and
        the embedding of edge position in the path.

    maximum_supported_path_length: `int`, optional (default=10)
        This value indicates the size of `self.codebook`, and MUST be larger than
        the following line in the dataset transform:

        ```python
        EncodeNode2NodeShortestPathFeatureTrajectory(
            max_length_considered=4, # <-
            feature_position_offset=4
        )
        ```
    """

    def __init__(
            self,
            num_heads: int,
            code_dim: int,
            maximum_supported_path_length: int,  # has to be larger than the corresponding transform
    ):
        super(PathTrajectoryEncodingAttentionBias, self).__init__()

        # - preparing the weights for these
        self.num_head = num_heads
        self.code_dim = code_dim
        self.edge_emb = torch.nn.Embedding(30, code_dim, padding_idx=0)
        self.codebook = torch.nn.Embedding(maximum_supported_path_length, code_dim * num_heads)
        self.maximum_supported_path_length = maximum_supported_path_length

    def encode_path_edge_type(self, shortest_path_feature_trajectory: torch.Tensor) -> torch.Tensor:
        """
        A simple mapping of the 3-dimensional edge feature in PCQM2Mv2 to a vector of dim `num_head` from the edge
        feature vector of dim `3`. The idea is that they decide to have the $d_E$ (equation 7 of the Graphormer paper)
        to be `num_heads`.

        Parameters
        ----------
        shortest_path_feature_trajectory: `torch.Tensor`, required
            `dim=(batch_size, max_node_number, max_node_number, L, feature_dim=3)`

        Returns
        -----------
        `torch.Tensor`:
            `dim=(batch_size, max_node_number, max_node_number, L, num_head)`
        """
        return torch.mean(
            self.edge_emb(
                shortest_path_feature_trajectory,
                # `dim=(batch_size, max_node_number, max_node_number, L, feature_dim=3)`
            ),  # after selecting embeddings:
            # `dim=(batch_size, max_node_number, max_node_number, L, feature_dim=3, code_dim)`
            dim=-2
        )  # after the mean reduction over edge feature positions:
        # `dim=(batch_size, max_node_number, max_node_number, L, code_dim)`

    def compute_supplementary_reps(self, *args, **kwargs):
        """no effect"""
        return 0

    def forward(
            self,
            shortest_path_feature_trajectory: torch.Tensor,
            shortest_path_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        shortest_path_feature_trajectory: `torch.Tensor`, required
            `dim=(batch_size, max_node_number, max_node_number, L, feature_dim=3)`

        shortest_path_lengths: `torch.Tensor`, required
            The shortest path lengths of `dim=(batch_size, max_node_number, max_node_number)`.

        Returns
        ----------
        `torch.Tensor`:
            The attention bias of `dim=batch_size, num_heads, max_num_nodes, max_num_nodes` corresponding
            to graphormer's path embedding
        """
        batch_size, max_num_nodes, _, max_distance_length, _ = shortest_path_feature_trajectory.size()

        # - with the addition of 1 (and given that 0 is considered padding), the paddings wont contribute
        # `dim=(batch_size, max_node_number, max_node_number, L, code_dim)`
        shortest_path_feature_trajectory = self.encode_path_edge_type(
            shortest_path_feature_trajectory=1 + shortest_path_feature_trajectory)

        # `dim=(L, batch_size, max_node_number, max_node_number, num_head)`
        shortest_path_feature_trajectory = shortest_path_feature_trajectory.permute(3, 0, 1, 2, 4).view(
            max_distance_length, -1, self.code_dim)

        # - now for each attention head, the idea is that for each edge number in the path, we
        # have a different path
        # `dim=(batch_size, max_node_number, max_node_number, num_head)`
        path_reps = torch.bmm(
            shortest_path_feature_trajectory,
            self.codebook.weight[:max_distance_length, :].view(max_distance_length, self.code_dim, self.num_head)
        ).reshape(
            max_distance_length, batch_size, max_num_nodes, max_num_nodes, self.num_head
        ).permute(1, 2, 3, 0, 4).sum(-2)  # b, n, n, h

        # - normalizing the result by path length
        path_reps = path_reps / shortest_path_lengths.clip(min=1).float().unsqueeze(-1)
        path_reps = path_reps.permute(0, 3, 1, 2).contiguous()  # b, h, n, n

        return path_reps
