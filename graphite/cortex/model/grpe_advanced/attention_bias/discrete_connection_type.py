from .base import AttentionBiasBase
import torch
import torch.nn


class DiscreteConnectionTypeEmbeddingAttentionBias(AttentionBiasBase):
    """
    Corresponds to the biases such as $\text{bias}^{\text{edge}}$ in the GRPE work.

    Parameters
    ----------
    model_dim: `int`, required
    num_heads: `int`, required
    num_connection_types: `int`, required
    """

    def __init__(
            self,
            model_dim: int,
            num_heads: int,
            num_connection_types: int,
    ):
        """constructor"""
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.num_connection_types = num_connection_types
        super(DiscreteConnectionTypeEmbeddingAttentionBias, self).__init__()

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
            edge_types: torch.LongTensor
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

        Returns
        ----------
        `torch.Tensor`:
            The additional bias term of `dim=(batch_size, num_heads, num_nodes, num_nodes)`
        """

        batch_size, num_heads, num_nodes, rep_dim = queries.size()

        E_q = self.connection_type_embedding_queries.weight  # num_connection_types, model_dim
        E_k = self.connection_type_embedding_keys.weight  # num_connection_types, model_dim

        # qE.size() == (b, h, n, num_connection_types)
        qE = torch.matmul(queries, E_q.view(1, self.num_connection_types, num_heads, rep_dim).permute(0, 2, 3,
                                                                                                      1))
        qE = torch.gather(
            qE,
            dim=3,
            index=edge_types.unsqueeze(1).repeat(1, num_heads, 1, 1)  # b, h, n, n
        )  # b, h, n, n

        kE = torch.matmul(
            keys,
            E_k.view(
                1,
                self.num_connection_types,
                num_heads,
                rep_dim
            ).permute(0, 2, 3, 1))  # (b, h, n, num_connection_types)
        kE = torch.gather(
            kE,
            dim=3,
            index=edge_types.unsqueeze(1).repeat(1, num_heads, 1, 1)  # b, h, n, n
        )  # b, h, n, n

        return qE + kE


class DiscreteConnectionTypeEmbeddingAttentionBiasComplementaryValues(torch.nn.Module):
    """
    Corresponds to the biases such as $\text{bias}^{\text{edge}}$ in the GRPE work.

    Parameters
    ----------
    model_dim: `int`, required
    num_heads: `int`, required
    num_connection_types: `int`, required
    """

    def __init__(
            self,
            model_dim: int,
            num_heads: int,
            num_connection_types: int,
    ):
        """constructor"""
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.num_connection_types = num_connection_types
        super(DiscreteConnectionTypeEmbeddingAttentionBiasComplementaryValues, self).__init__()

        self.connection_type_embedding_values = torch.nn.Embedding(
            num_connection_types,
            model_dim,
        )

    def forward(
            self,
            attention_weights: torch.Tensor,
            edge_types: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Computing the complementary values to be added to the original output representations
        which resulted from `a @ v`.

        Parameters
        ----------
        attention_weights: `torch.Tensor`, required
            attention weights AFTER softmax of `dim=(b, h, n, n)`

        edge_types: `torch.LongTensor`, required
            The discrete edge types, which is of dimension `batch_size, max_num_nodes, max_num_nodes`.
            If this is being used, we need to note that in the `edge_types` which are rendered (originally), the indices
            can go from 0 to 23. In this, given that we have the special edge types too.

        Returns
        ----------
        `torch.Tensor`:
            The outputs of dim `batch_size, num_heads, num_nodes, rep_dim`
        """
        # - sizes
        batch_size, num_heads, num_nodes, num_nodes = attention_weights.size()

        # - placeholder
        supplementary_value = torch.zeros(
            (batch_size, num_heads, num_nodes, self.num_connection_types)
        ).to(attention_weights.device).float()

        # - note that edge_types is of `dim=batch_size, num_nodes, num_nodes`, therefore,
        # upon repeating we will have `dim=batch_size, num_heads, num_nodes, num_nodes` with each
        # value being the connection type. Note that for each node, each edge type (e.g. $\phi(i,j)$),
        # we are summing all the attention values of nodes that have that edge-type between them
        # and our query node.
        supplementary_value = torch.scatter_add(
            supplementary_value, 3, edge_types.unsqueeze(1).repeat(1, num_heads, 1, 1), attention_weights
        )  # dim=batch_size, num_heads, num_nodes, num_connections

        E_v = self.connection_type_embedding_values.weight  # num_connection_types, model_dim

        outputs = torch.matmul(
            supplementary_value,  # b, h, n, num_connections
            E_v.view(1, self.num_connection_types, num_heads, -1).permute(0, 2, 1, 3)  # 1, h, num_connections, rep_dim
        )  # dim: b, h, n, rep_dim=model_dim//h

        return outputs
