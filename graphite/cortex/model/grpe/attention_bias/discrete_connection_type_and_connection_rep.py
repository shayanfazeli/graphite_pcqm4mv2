from .base import AttentionBiasBase
import torch
import torch.nn
from typing import Tuple, List
from .discrete_connection_type import DiscreteConnectionTypeEmbeddingAttentionBias


class DiscreteConnectionTypeEmbeddingPlusConnectionRepresentationAttentionBias(DiscreteConnectionTypeEmbeddingAttentionBias):
    """
    Corresponds to the biases such as $\text{bias}^{\text{edge}}$ in the GRPE work.
    """
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super(DiscreteConnectionTypeEmbeddingPlusConnectionRepresentationAttentionBias, self).__init__(*args, **kwargs)
        self.projection_query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.LayerNorm(self.model_dim // self.num_heads),
            torch.nn.Linear(self.model_dim // self.num_heads, self.model_dim // self.num_heads),
        )
        self.projection_key = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.LayerNorm(self.model_dim // self.num_heads),
            torch.nn.Linear(self.model_dim // self.num_heads, self.model_dim // self.num_heads),
        )

    def forward(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            edge_types: torch.LongTensor,
            edges: List[Tuple[torch.LongTensor, torch.LongTensor]],
            edge_reps: List[torch.Tensor],
            layer_index: int = None
    ):
        """
        Parameters
        ----------
        queries: `torch.Tensor`, required
            `dim=(batch_size, num_heads, num_nodes, rep_dim)`

        keys: `torch.Tensor`, required
            `dim=(batch_size, num_heads, num_nodes, rep_dim)`

        edge_types: `torch.LongTensor`, required
            The discrete edge types, which is of dimension `batch_size, max_num_nodes, max_num_nodes`.
            If this is being used, we need to note that in the `edge_types` which are rendered (originally), the indices
            can go from 0 to 23. In this, given that we have the special edge types too.

        edge_reps: `torch.Tensor`, required
            The representations of `dim=(batch_size, num_edges, model_dim)`
        """
        batch_size, num_heads, num_nodes, rep_dim = queries.size()
        assert edge_reps[0].shape[-1] == num_heads * rep_dim
        assert len(edge_reps) == batch_size
        bias_by_types = super().forward(
            queries=queries,
            keys=keys,
            edge_types=edge_types,
            layer_index=layer_index
        )  # dim=b,h,n,n

        for i, ((u, v), z) in enumerate(zip(edges, edge_reps)):
            edge_biases_query = torch.sum(self.projection_query(
                z.view(u.shape[0], num_heads, rep_dim)
            ).transpose(0, 1) * queries[i, :, u, :], dim=-1)  # m, h, rep_dim X h, m, rep_dim
            edge_biases_key = torch.sum(self.projection_key(
                z.view(u.shape[0], num_heads, rep_dim)
            ).transpose(0, 1) * keys[i, :, v, :], dim=-1)  # m, h, rep_dim X h, m, rep_dim
            bias_by_types[i, :, u, v] = bias_by_types[i, :, u, v] + edge_biases_query + edge_biases_key

        return bias_by_types
