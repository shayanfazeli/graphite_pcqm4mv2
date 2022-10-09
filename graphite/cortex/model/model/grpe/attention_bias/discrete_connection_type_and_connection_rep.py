from overrides import overrides
import torch
import torch.nn
from typing import Tuple, List
from .discrete_connection_type import DiscreteConnectionTypeEmbeddingAttentionBias


class DiscreteConnectionTypeEmbeddingPlusConnectionRepresentationAttentionBias(
    DiscreteConnectionTypeEmbeddingAttentionBias
):
    """
    In addition to the usual `tokenized` representation and biases in grpe (which are implemented
    in :cls:`DiscreteConnectionTypeEmbeddingAttentionBias`, this module can consider continuous edge reps as well.

    The way it works is that, in addition to the represented edge types, we also have `edge_reps` which
    are of `dim=(batch_size, num_edges, model_dim)`, along with COO format edges. They will be projected
    (since they are not long integers to be embedded) and that projection for query and key will
    be used to guide the bias, meaning that we would have a $qE_{q} + kE_{k}$ this way as well.

    One potential use for this could be to combine the 2d reps with the 3d-based
    continuous edge representations for a 3d-molecule representation.
    """
    def __init__(
            self,
            *args,
            **kwargs
    ):
        """constructor"""
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

    def compute_supplementary_reps(
            self,
            attention_weights: torch.Tensor,
            edge_types: torch.LongTensor,
            values: torch.Tensor = None,
    ) -> torch.Tensor:
        pass
    #todo: implement this

    def forward(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            edge_types: torch.LongTensor,
            edges: List[Tuple[torch.LongTensor, torch.LongTensor]],
            edge_reps: List[torch.Tensor]
    ) -> torch.Tensor:
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

        edges: `List[Tuple[torch.LongTensor, torch.LongTensor]]`, required
            The coordinate representation of edges (COO format)

        edge_reps: `torch.Tensor`, required
            The representations of `dim=(batch_size, num_edges, model_dim)`
        """
        batch_size, num_heads, num_nodes, rep_dim = queries.size()
        assert edge_reps[0].shape[-1] == num_heads * rep_dim
        assert len(edge_reps) == batch_size
        bias_by_types = super().forward(
            queries=queries,
            keys=keys,
            edge_types=edge_types
        )  # dim=b,h,n,n

        for i, ((u, v), z) in enumerate(zip(edges, edge_reps)):
            edge_biases_query = torch.bmm(
                self.projection_query(z.view(u.shape[0], num_heads, rep_dim)).transpose(0, 1),  # num_head, num_edges, rep_dim
                queries[i, :, u, :].transpose(-1, -2),  # num_heads, rep_dim, num_edges
            )  # num_heads, num_edges, num_edges
            edge_biases_key = torch.bmm(
                self.projection_key(z.view(u.shape[0], num_heads, rep_dim)).transpose(0, 1),  # num_head, num_edges, rep_dim
                keys[i, :, v, :].transpose(-1, -2))  # num_heads, num_edges, num_edges
            bias_by_types[i, :, u, v] = bias_by_types[i, :, u, v] + edge_biases_query + edge_biases_key

        return bias_by_types
