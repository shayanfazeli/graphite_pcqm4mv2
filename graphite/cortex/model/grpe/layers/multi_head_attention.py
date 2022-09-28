import torch
import torch.nn


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
        self.linear_q = torch.nn.Linear(model_dim, num_heads * rep_dim)
        self.linear_k = torch.nn.Linear(model_dim, num_heads * rep_dim)
        self.linear_v = torch.nn.Linear(model_dim, num_heads * rep_dim)
        self.att_dropout = torch.nn.Dropout(attention_dropout)

        self.output_layer = torch.nn.Linear(num_heads * self.rep_dim, model_dim)

    def forward(
        self,
        node_reps: torch.Tensor,
        edge_bias_guide: torch.nn.Module,
        path_length_bias_guide: torch.nn.Module,
        distance: torch.LongTensor,
        connection_reps: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_reps: `torch.Tensor`, required
            The node representations (padded) of dim=`batch_size, max_node_count, model_dim`.

        edge_bias_guide: `torch.nn.Module`, required
            The bias guide for including *edge type representations*.

        path_length_bias_guide: `torch.nn.Module`, required
            The bias guide for including *shortest path length type representations*.

        distance: `torch.LongTensor`, required
            The shortest path distance types (meaning that the task distance and padding is included too).

        connection_reps: `torch.Tensor`, required
            The edge types between nodes of `dim=(batch_size, max_node_count, max_node_count)`

        mask: `torch.Tensor`, optional(default=None)
            The sequence mask `dim=(batch_size, max_node_count)`.

        Returns
        ----------
        `torch.Tensor`:
            The representations of dim `batch_size, max_node_count, model_dim`.
        """
        # - input dim
        input_dims = node_reps.size()

        # - rep dim
        d_k = self.rep_dim
        d_v = self.rep_dim
        batch_size = node_reps.size(0)

        # - getting the queries, keys, and values
        # dims: (batch_size, num_heads, num_items(num_nodes), rep_dim
        queries = self.linear_q(node_reps).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)
        keys = self.linear_k(node_reps).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)
        values = self.linear_v(node_reps).view(batch_size, -1, self.num_heads, d_v).transpose(1, 2)

        # - the core attention `a`
        a = torch.matmul(queries, keys.transpose(2, 3))  # b, h, n, n

        # - the edge type bias
        a = a + edge_bias_guide(
            queries=queries,
            keys=keys,
            edge_types=connection_reps
        )

        # - the shortest path length type bias
        a = a + path_length_bias_guide(
            queries=queries,
            keys=keys,
            edge_types=distance
        )

        # - scaling, masking, and softmax
        a = a * self.scale
        if mask is not None:
            a = a.masked_fill(
                mask.view(mask.shape[0], 1, 1, mask.shape[1]), float("-inf")
            )
        # a = torch.softmax(a.clip(min=-10000), dim=3) # in case of inf causing nans
        a = torch.softmax(a, dim=3)

        # - attention dropout
        a = self.att_dropout(a)

        z = edge_bias_guide.compute_supplementary_reps(
            attention_weights=a,
            edge_types=connection_reps,
            values=values
        )

        z = z + path_length_bias_guide.compute_supplementary_reps(
            attention_weights=a,
            edge_types=distance,
            values=None
        )

        z = z.transpose(1, 2).contiguous()  # b, n, h, rep_dim
        z = z.view(batch_size, -1, self.num_heads * d_v) # b, n, model_dim

        z = self.output_layer(z)
        assert z.size() == input_dims
        return z
