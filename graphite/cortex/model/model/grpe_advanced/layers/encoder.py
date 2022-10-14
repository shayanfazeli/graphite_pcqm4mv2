from typing import Tuple
import torch
import torch.nn
from .multi_head_attention import MultiHeadAttention
from .feed_forward import FeedForwardNetwork


class EncoderLayer(torch.nn.Module):
    """
    The encoder layer

    Parameters
    ----------
    model_dim: `int`, required
    feedforward_dim: `int`, required
    dropout_rate: `float`, required
    attention_dropout_rate: `float`, required
    num_heads: `int`,  required
    """
    def __init__(
            self,
            model_dim: int,
            feedforward_dim: int,
            dropout_rate: float,
            attention_dropout_rate: float,
            num_heads: int
    ):
        """constructor"""
        super(EncoderLayer, self).__init__()

        # - layer norm
        self.self_attention_norm = torch.nn.LayerNorm(model_dim)

        # - MHA
        self.self_attention = MultiHeadAttention(
            model_dim,
            attention_dropout_rate,
            num_heads,
        )

        # - dropout for attention
        self.self_attention_dropout = torch.nn.Dropout(dropout_rate)

        # - layer norm and ffn modules
        self.ffn_norm = torch.nn.LayerNorm(model_dim)
        self.ffn = FeedForwardNetwork(model_dim, feedforward_dim, dropout_rate)
        self.ffn_dropout = torch.nn.Dropout(dropout_rate)

    def forward(
        self,
        node_reps: torch.Tensor,
        distance: torch.Tensor,
        connection_reps: torch.Tensor,
        edge_bias_guide: torch.nn.Module,
        shortest_path_length_bias_guide: torch.nn.Module,
        shortest_path_bias_guide: torch.nn.Module,
        shortest_path_feature_trajectory: torch.Tensor,
        edge_bias_guide_complementary_values: torch.nn.Module,
        shortest_path_length_bias_guide_complementary_values: torch.nn.Module,
        mask: torch.Tensor,
        toeplitz: Tuple[torch.nn.parameter.Parameter, torch.nn.parameter.Parameter]
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_reps: `torch.Tensor`, required
            The node representations (padded) of dim=`batch_size, max_node_count, model_dim`.

        edge_bias_guide: `torch.nn.Module`, required
            The bias guide for including *edge type representations*.

        shortest_path_length_bias_guide: `torch.nn.Module`, required
            The bias guide for including *shortest path length type representations*.

        edge_bias_guide_complementary_values: `torch.nn.Module`, required
            The module for computing the complementary outputs for the corresponding attention modifier

        shortest_path_length_bias_guide_complementary_values: `torch.nn.Module`, required
            The module for computing the complementary outputs for the corresponding attention modifier

        distance: `torch.LongTensor`, required
            The shortest path distance types (meaning that the task distance and padding is included too).

        connection_reps: `torch.Tensor`, required
            The edge types between nodes of `dim=(batch_size, max_node_count, max_node_count)`

        shortest_path_bias_guide: `torch.nn.Module`, required
            The bias of graphormer for shortest path encodings

        shortest_path_feature_trajectory: `torch.Tensor`, required
            The shortest path feature trajectory composed of offsetted edge features (discrete)
            of `dim=(batch_size, num_nodes, num_nodes, path_length, 3)`

        mask: `torch.Tensor`, optional(default=None)
            The sequence mask `dim=(batch_size, max_node_count)`.

        toeplitz: `Tuple[torch.nn.parameter.Parameter, torch.nn.parameter.Parameter]`, required
            If provided, it will be the row and column parameter for toeplitz-based modification
            of post-softmax attention weights.

        Returns
        ----------
        `torch.Tensor`:
            The representations of dim `batch_size, max_node_count, model_dim`.
        """
        reps = self.self_attention_norm(node_reps)
        reps = self.self_attention(
            node_reps=reps,
            edge_bias_guide=edge_bias_guide,
            shortest_path_length_bias_guide=shortest_path_length_bias_guide,
            shortest_path_bias_guide=shortest_path_bias_guide,
            distance=distance,
            connection_reps=connection_reps,
            shortest_path_feature_trajectory=shortest_path_feature_trajectory,
            mask=mask,
            edge_bias_guide_complementary_values=edge_bias_guide_complementary_values,
            shortest_path_length_bias_guide_complementary_values=shortest_path_length_bias_guide_complementary_values,
            toeplitz=toeplitz
        )
        reps = self.self_attention_dropout(reps)
        node_reps = node_reps + reps

        reps = self.ffn_norm(node_reps)
        reps = self.ffn(reps)
        reps = self.ffn_dropout(reps)
        node_reps = node_reps + reps
        return node_reps
