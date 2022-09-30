from typing import Dict, List, Any
import copy
import torch
import torch.nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from .attention_bias import DiscreteConnectionTypeEmbeddingAttentionBias
from .layers.encoder import EncoderLayer
from ..encoders.connection.pcqm4mv2 import EmbedPCQM4Mv2ShortestPathLengthType, EmbedPCQM4Mv2EdgeType
from ..encoders.node.pcqm4mv2 import EmbedPCQM4Mv2NodeFeatures


def get_mask_from_sequence_lengths(
    sequence_lengths: torch.Tensor, max_length: int
) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.  For example, if
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`. (NEGATION OF THIS IS DONE)
    We require `max_length` here instead of just computing it from the input `sequence_lengths`
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return ~(sequence_lengths.unsqueeze(1) >= range_tensor)


class GraphRelativePositionalEncodingNetwork(torch.nn.Module):
    """
    The :cls:`GraphRelativePositionalEncodingNetwork` is our reimplementation of the GRPE network
    for graph representation [original repository: [link](https://github.com/lenscloth/GRPE/)].

    Parameters
    ----------
    model_dimension: `int`, optional(default=512)
        The model dimension

    number_of_heads: `int`, optional(default=8)
        The attention parameter

    number_of_layers: `int`, optional(default=6)
        The number of encoder layers

    feedforward_dimension: `int`, optional(default=2048)
        The dimension of feedforward component in the transformer's feedforward

    dropout: `float`, optional(default=0.1)
        The dropout for the encoder's feedforward

    attention_dropout: `float`, optional(default=0.1)
        The attention dropout

    shortest_path_length_upperbound: `int`, optional(default=256)
        The maximum path length to be considered (this value + 5 will be the length of the corresponding
        embedding codebook (5 is just in case, in reality, there is unreachable and task distance).

    perturbation: `float`, optional(default=0.0)
        Model perturbation rate, unused at the moment.

    independent_layer_embeddings: `bool`, optional(default=False)
        Whether to create a separate bias guide per layer
    """

    def __init__(
            self,
            model_dimension: int = 512,
            number_of_heads: int = 8,
            number_of_layers: int = 6,
            feedforward_dimension: int = 2048,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            shortest_path_length_upperbound: int = 256,
            perturbation: float = 0.0,
            independent_layer_embeddings: bool = False,
    ):
        """constructor"""
        super(GraphRelativePositionalEncodingNetwork, self).__init__()
        self.node_encoder = EmbedPCQM4Mv2NodeFeatures(model_dim=model_dimension)
        self.perturbation = perturbation
        assert perturbation == 0
        self.shortest_path_length_upperbound = shortest_path_length_upperbound
        self.model_dim = model_dimension
        self.independent_layer_embeddings = independent_layer_embeddings

        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads

        if not self.independent_layer_embeddings:
            self.attention_bias_edge = DiscreteConnectionTypeEmbeddingAttentionBias(
                model_dim=model_dimension,
                num_heads=number_of_heads,
                num_connection_types=31,
            )
            self.attention_bias_shortest_path = DiscreteConnectionTypeEmbeddingAttentionBias(
                model_dim=model_dimension,
                num_heads=number_of_heads,
                num_connection_types=self.shortest_path_length_upperbound+5,
            )
        else:
            self.attention_bias_edge = {i: DiscreteConnectionTypeEmbeddingAttentionBias(
                model_dim=model_dimension,
                num_heads=number_of_heads,
                num_connection_types=31,
            ) for i in range(self.number_of_layers)}
            self.attention_bias_shortest_path = {i: DiscreteConnectionTypeEmbeddingAttentionBias(
                model_dim=model_dimension,
                num_heads=number_of_heads,
                num_connection_types=self.shortest_path_length_upperbound+5,
            ) for i in range(self.number_of_layers)}

        self.layered_stem = torch.nn.ModuleList([
            EncoderLayer(
                model_dim=model_dimension,
                feedforward_dim=feedforward_dimension,
                dropout_rate=dropout,
                attention_dropout_rate=attention_dropout,
                num_heads=number_of_heads,
            ) for _ in range(self.number_of_layers)
        ])

    def forward(
            self,
            batch: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        batch: `Dict[str, Any]`, required
            A collated batch of the dataset

        Returns
        ----------
        `torch.Tensor`:
            The graph embeddings of `dim=(batch_size, model_dim)`.
        """
        # - getting the batched graphs
        batched_graph = batch['graphs']

        # - getting the encoded node types (with the offset applied)
        node_type = batch['node_type']

        # - connection types (edge types tokenized + no edge, self edge, and task edge
        node2node_connection_types = batch['node2node_connection_type']
        shortest_path_length_types = batch['node2node_shortest_path_length_type']
        device = node_type.device

        # - performing the representation step for nodes
        node_features = self.node_encoder(node_type)

        # - getting the node counts (including the task node)
        graph_node_counts = [batched_graph[i].num_nodes for i in range(len(batched_graph))]
        graph_node_counts = torch.tensor(graph_node_counts).to(device)
        max_node_count = max(graph_node_counts)
        mask = get_mask_from_sequence_lengths(
            graph_node_counts,
            max_length=max_node_count
        )

        # - performing the layer by layer encoding
        for i, encoder_layer in enumerate(self.layered_stem):
            if self.independent_layer_embeddings:
                node_features = encoder_layer(
                    node_features,
                    distance=shortest_path_length_types,
                    connection_reps=node2node_connection_types,
                    edge_bias_guide=self.attention_bias_edge[i],
                    path_length_bias_guide=self.attention_bias_shortest_path[i],
                    mask=mask,
                )
            else:
                node_features = encoder_layer(
                    node_features,
                    distance=shortest_path_length_types,
                    connection_reps=node2node_connection_types,
                    edge_bias_guide=self.attention_bias_edge,
                    path_length_bias_guide=self.attention_bias_shortest_path,
                    mask=mask,
                )

        # - gathering the representations for the task node as graph embeddings
        last_node_indices = graph_node_counts - 1
        graph_reps = torch.gather(node_features, 1,
                                  last_node_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.model_dim)).squeeze(1)

        return graph_reps
