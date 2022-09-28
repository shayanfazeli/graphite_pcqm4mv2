from graphite.cortex.model.grpe.attention_bias.base import AttentionBiasBase
import torch
import torch.nn


class PathTrajectoryEncodingAttentionBias(AttentionBiasBase):
    """
    Corresponds to the path encoding of the graphormer work.

    Please note that the idea here depends on scalar encoding of
    edge types.

    Regarding the forward variables, in the current collation, we can have:

    ```
    %%time
    shortest_path_feature_trajectory = g[0].ndata['shortest_path_feature_trajectory']
    shortest_path_feature_trajectory = torch.cat((shortest_path_feature_trajectory, -torch.ones((shortest_path_feature_trajectory.size(0), 1, shortest_path_feature_trajectory.size(2), shortest_path_feature_trajectory.size(3)))), dim=1)
    shortest_path_feature_trajectory = torch.stack([shortest_path_feature_trajectory for _ in range(32)])
    shortest_path_lengths = g[0].ndata['node2node_shortest_path_length_type']
    shortest_path_lengths = torch.stack([shortest_path_lengths for _ in range(32)])
    attn_bias(shortest_path_feature_trajectory=shortest_path_feature_trajectory.to(device),shortest_path_lengths=shortest_path_lengths.to(device))
    ```

    Which will take around 15ms to complete.
    """

    def __init__(
            self,
            num_head: int,
            maximum_supported_path_length: int = 10,
    ):
        super(PathTrajectoryEncodingAttentionBias, self).__init__()

        # - preparing the weights for these
        self.num_head = num_head
        self.edge_emb = torch.nn.Embedding(3, num_head, padding_idx=0)
        self.codebook = torch.nn.Embedding(maximum_supported_path_length, num_head * num_head)
        self.maximum_supported_path_length = maximum_supported_path_length

    def encode_path_edge_type(self, shortest_path_feature_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        shortest_path_feature_trajectory: `torch.Tensor`, required
            `dim=(batch_size, max_node_number, max_node_number, L, feature_dim=3)`

        Returns
        -----------
        `torch.Tensor`:
            `dim=(batch_size, max_node_number, max_node_number, L, num_head)`
        """
        return torch.matmul(shortest_path_feature_trajectory, self.edge_emb.weight)

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

        # `dim=(batch_size, max_node_number, max_node_number, L, num_head)`
        shortest_path_feature_trajectory = self.encode_path_edge_type(
            shortest_path_feature_trajectory=1 + shortest_path_feature_trajectory)

        # `dim=(L, batch_size, max_node_number, max_node_number, num_head)`
        shortest_path_feature_trajectory = shortest_path_feature_trajectory.permute(3, 0, 1, 2, 4).view(
            max_distance_length, -1, self.num_head)

        # `dim=(batch_size, max_node_number, max_node_number, num_head)`
        path_reps = torch.bmm(
            shortest_path_feature_trajectory,
            self.codebook.weight[:max_distance_length, :].view(max_distance_length, self.num_head, self.num_head)
        ).reshape(
            max_distance_length, batch_size, max_num_nodes, max_num_nodes, self.num_head
        ).permute(1, 2, 3, 0, 4).sum(-2)  # b, n, n, h

        path_reps = path_reps / (1 + shortest_path_lengths).float().unsqueeze(-1)
        path_reps = path_reps.permute(0, 3, 1, 2).contiguous()  # b, h, n, n

        return path_reps
