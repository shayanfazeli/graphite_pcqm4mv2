from .base import AttentionBiasBase
import torch
import torch.nn


class DiscreteConnectionTypeEmbeddingAttentionBias(AttentionBiasBase):
    """
    Corresponds to the biases such as $\text{bias}^{\text{edge}}$ in the GRPE work.
    """
    def __init__(
            self,
            num_layers: int,
            independent_weights_per_layer: bool,
            model_dim: int,
            num_heads: int,
            num_connection_types: int,
    ):
        super(DiscreteConnectionTypeEmbeddingAttentionBias, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.independent_weights_per_layer = independent_weights_per_layer
        self.model_dim = model_dim
        self.num_connection_types = num_connection_types

        if self.independent_weights_per_layer:
            self.connection_type_embedding_queries = {i: torch.nn.Embedding(
                num_connection_types,
                model_dim,
            ) for i in range(self.num_layers)}
            self.connection_type_embedding_keys = {i: torch.nn.Embedding(
                num_connection_types,
                model_dim,
            ) for i in range(self.num_layers)}
        else:
            self.connection_type_embedding_queries = torch.nn.Embedding(
                num_connection_types,
                model_dim,
            )
            self.connection_type_embedding_keys = torch.nn.Embedding(
                num_connection_types,
                model_dim,
            )

    def forward(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            edge_types: torch.LongTensor,
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
        """
        if self.independent_weights_per_layer:
            assert layer_index is not None

        batch_size, num_heads, num_nodes, rep_dim = queries.size()
        
        if self.independent_weights_per_layer:
            E_q = self.connection_type_embedding_queries[layer_index].weight
            E_k = self.connection_type_embedding_keys[layer_index].weight
        else:
            E_q = self.connection_type_embedding_queries.weight
            E_k = self.connection_type_embedding_keys.weight

        qE = torch.matmul(queries, E_q.view(1, self.num_connection_types, num_heads, rep_dim).permute(0,2,3,1))  # (b, h, n, num_connection_types)
        qE = torch.gather(
            qE, 
            dim=3, 
            index=edge_types.unsqueeze(1).repeat(1, num_heads, 1, 1)  # b, h, n, n
        )  # b, h, n, n

        kE = torch.matmul(keys,
                          E_k.view(1, self.num_connection_types, num_heads, rep_dim).permute(0, 2, 3,
                                                                                                            1))  # (b, h, n, num_connection_types)
        kE = torch.gather(
            kE,
            dim=3,
            index=edge_types.unsqueeze(1).repeat(1, num_heads, 1, 1)  # b, h, n, n
        )  # b, h, n, n
        
        return qE + kE
