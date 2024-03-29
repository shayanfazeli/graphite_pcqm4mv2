from typing import Tuple
import copy
import torch
import torch.nn

from graphite.utilities.miscellaneous import toeplitz_multihead


class MultiHeadAttention(torch.nn.Module):
    """
    The re-implementation of the specific multi-head attention
    used in GRPE.

    Parameters
    ----------
    model_dim: `int`, required
        Model dimension which will be used as hidden size in here.

    attention_dropout: `float`, required
        The attention dropout

    num_heads: `int`, required
        number of attention heads
    """
    def __init__(
            self,
            model_dim: int,
            attention_dropout: float,
            num_heads: int
    ):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.rep_dim = rep_dim = model_dim // num_heads
        self.scale = rep_dim ** -0.5
        # self.linear_q = torch.nn.Linear(model_dim, num_heads * rep_dim)
        # self.linear_k = torch.nn.Linear(model_dim, num_heads * rep_dim)
        # self.linear_v = torch.nn.Linear(model_dim, num_heads * rep_dim)

        self.linear_qkv = torch.nn.Linear(model_dim, 3 * num_heads * rep_dim)
        self.att_dropout = torch.nn.Dropout(attention_dropout)

        self.output_layer = torch.nn.Linear(num_heads * self.rep_dim, model_dim)

    def compute_attention_weights(
            self,
            node_reps: torch.Tensor,
    ):
        # - rep dim
        d_k = self.rep_dim
        d_v = self.rep_dim
        assert d_k == d_v
        batch_size = node_reps.size(0)

        # - getting the queries, keys, and values
        # dims: (batch_size, num_heads, num_items(num_nodes), rep_dim
        # queries = self.linear_q(node_reps).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)
        # keys = self.linear_k(node_reps).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)
        # values = self.linear_v(node_reps).view(batch_size, -1, self.num_heads, d_v).transpose(1, 2)
        queries, keys, values = [e.squeeze(-3).transpose(1, 2) for e in
                                 torch.split(self.linear_qkv(node_reps).view(batch_size, -1, 3, self.num_heads, d_k), 1,
                                             dim=-3)]

        # - the core attention `a`
        a = torch.matmul(queries, keys.transpose(2, 3))  # b, h, n, n

        return a, (queries, keys, values)

    def forward(
        self,
        node_reps: torch.Tensor,
        edge_bias_guide: torch.nn.Module,
        shortest_path_length_bias_guide: torch.nn.Module,
        shortest_path_bias_guide: torch.nn.Module,
        distance: torch.LongTensor,
        connection_reps: torch.Tensor,
        shortest_path_feature_trajectory: torch.Tensor,
        edge_bias_guide_complementary_values: torch.nn.Module,
        shortest_path_length_bias_guide_complementary_values: torch.nn.Module,
        mask: torch.Tensor = None,
        toeplitz: Tuple[torch.nn.parameter.Parameter, torch.nn.parameter.Parameter] = None
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

        shortest_path_bias_guide: `torch.nn.Module`, required
            The bias of graphormer for shortest path encodings

        shortest_path_feature_trajectory: `torch.Tensor`, required
            The shortest path feature trajectory composed of offsetted edge features (discrete)
            of `dim=(batch_size, num_nodes, num_nodes, path_length, 3)`

        edge_bias_guide_complementary_values: `torch.nn.Module`, required
            The module for computing the complementary outputs for the corresponding attention modifier

        shortest_path_length_bias_guide_complementary_values: `torch.nn.Module`, required
            The module for computing the complementary outputs for the corresponding attention modifier

        distance: `torch.LongTensor`, required
            The shortest path distance types (meaning that the task distance and padding is included too).

        connection_reps: `torch.Tensor`, required
            The edge types between nodes of `dim=(batch_size, max_node_count, max_node_count)`

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
        # - sizes
        input_dims = node_reps.size()
        batch_size = node_reps.size(0)
        d_k = self.rep_dim
        d_v = self.rep_dim

        # - computing the attention weights
        a, (queries, keys, values) = self.compute_attention_weights(node_reps=node_reps)

        # - the edge type bias
        if edge_bias_guide is not None:
            a = a + edge_bias_guide(
                queries=queries,
                keys=keys,
                edge_types=connection_reps
            )

        # - the shortest path length type bias
        if shortest_path_length_bias_guide is not None:
            a = a + shortest_path_length_bias_guide(
                queries=queries,
                keys=keys,
                edge_types=distance
            )

        # - the encoded shortest path trajectory bias
        if shortest_path_bias_guide is not None:
            a = a + shortest_path_bias_guide(
                shortest_path_feature_trajectory=shortest_path_feature_trajectory,
                shortest_path_lengths=distance,
            )

        # - scaling, masking, and softmax
        a = a * self.scale
        if mask is not None:
            a = a.masked_fill(
                mask.view(mask.shape[0], 1, 1, mask.shape[1]), -torch.inf
            )

        # - b, h, n, n
        a = torch.softmax(a, dim=3)

        if toeplitz is not None:
            n = a.shape[-1]
            a = a * toeplitz_multihead(r=toeplitz[0], c=toeplitz[1])[:, :n, :n].unsqueeze(0)

        # - attention dropout
        a = self.att_dropout(a)

        # - base attention outputs
        z = torch.matmul(a, values)

        # - adding the additional outputs for the biases, if any
        if edge_bias_guide is not None:
            if edge_bias_guide_complementary_values is not None:
                z = z + edge_bias_guide_complementary_values(
                    attention_weights=a,
                    edge_types=connection_reps,
                )

        if shortest_path_length_bias_guide is not None:
            if shortest_path_length_bias_guide_complementary_values is not None:
                z = z + shortest_path_length_bias_guide_complementary_values(
                    attention_weights=a,
                    edge_types=distance,
                )

        z = z.transpose(1, 2).contiguous()  # b, n, h, rep_dim
        z = z.view(batch_size, -1, self.num_heads * d_v) # b, n, model_dim

        z = self.output_layer(z)
        assert z.size() == input_dims
        return z
