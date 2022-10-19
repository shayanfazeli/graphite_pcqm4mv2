from typing import Dict, List, Any
import copy
import numpy
import torch
import torch.nn
from torch_geometric.data import Data, Batch
from graphite.utilities.masking.utilities import get_mask_from_sequence_lengths
from .attention_bias import DiscreteConnectionTypeEmbeddingAttentionBias, DiscreteConnectionTypeEmbeddingAttentionBiasComplementaryValues
from .attention_bias.path_feature_trajectory_encoding import PathTrajectoryEncodingAttentionBias
from .layers.encoder import EncoderLayer
from ..encoders.node.pcqm4mv2 import EmbedPCQM4Mv2NodeFeatures
import graphite.data.utilities.pcqm4mv2_meta as PCQM4MV2_METADATA


class GraphRelativePositionalEncodingNetworkAdvanced(torch.nn.Module):
    """
    The :csl:`GraphRelativePositionalEncodingNetworkAdvanced` is an extended versio nof
    :cls:`GraphRelativePositionalEncodingNetwork` which allows additional complementary modules to be involved.

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
        Node feature perturbation

    independent_layer_embeddings: `bool`, optional(default=False)
        Whether to create a separate bias guide per layer

    attention_biases: `List[str]`, optional (default=['edge','shortest_path_length','shortest_path'])
        You can choose which additional attention biases to be applied to the data.
        The following options are supported
            * `edge`: the edge type bias (based on encoded bond type in pcqm4mv2)
            * `shortest_path_length`: based on the encoded shortest path length (upperbounded and special values)
            * `shortest_path`: based on the shortest path encoding (introduced in graphormer)

    path_encoding_length_upperbound: `int`, optional (default=5)
        This parameter is related to the `shortest_path` attention bias. To be precise,
        this is dirrectly related to `max_length_considered` used in :cls:`EncodeNode2NodeShortestPathFeatureTrajectory`.

    path_encoding_code_dim: `int`, optional (default=None)
        The dimension of each embedding for encoding the shortest path trajectory. This parameter is related to
        the `shortest_path` attention bias. To be precise, this is dirrectly related to `max_length_considered` used
        in :cls:`EncodeNode2NodeShortestPathFeatureTrajectory`.

    encode_node_degree_centrality: `bool`, optional (default=False)
        To  perform degree  centrality encoding (per Graphormer's pipeline). Please note that normally
        this is not done in GRPE pipeline.

    node_degree_upperbound: `int`, optional (default=50)
        Upperbound for node degree centrality encoding

    toeplitz: `bool`, optional (default=False)
        If `True`, the parameterization of [this paper](https://arxiv.org/pdf/2205.13401.pdf) will
        be applied on the MHA layers.
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
            attention_biases: List[str] = [
                'edge',
                'shortest_path_length',
                'shortest_path'
            ],
            path_encoding_length_upperbound: int = 5,
            path_encoding_code_dim: int = None,
            encode_node_degree_centrality: bool = False,
            node_degree_upperbound: int = 50,
            toeplitz: bool = False
    ):
        """constructor"""
        super(GraphRelativePositionalEncodingNetworkAdvanced, self).__init__()
        self.node_encoder = EmbedPCQM4Mv2NodeFeatures(model_dim=model_dimension)
        self.perturbation = perturbation
        self.shortest_path_length_upperbound = shortest_path_length_upperbound
        self.model_dim = model_dimension
        self.independent_layer_embeddings = independent_layer_embeddings
        self.path_encoding_length_upperbound = path_encoding_length_upperbound
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.attention_biases = set(attention_biases)
        self.path_encoding_code_dim = path_encoding_code_dim if path_encoding_code_dim is not None else number_of_heads
        self.encode_node_degree_centrality = encode_node_degree_centrality
        if self.encode_node_degree_centrality:
            self.node_degree_centrality_encoder = EmbedPCQM4Mv2NodeFeatures(
                model_dim=model_dimension,
                codebook_length=node_degree_upperbound + 3,
                padding_idx=0)
            self.node_degree_upperbound = node_degree_upperbound

        if not self.independent_layer_embeddings:
            self.attention_bias_edge = DiscreteConnectionTypeEmbeddingAttentionBias(
                model_dim=model_dimension,
                num_heads=number_of_heads,
                num_connection_types=PCQM4MV2_METADATA.num_discrete_bond_types + 4,
            ) if 'edge' in self.attention_biases else None
            self.attention_bias_shortest_path_length = DiscreteConnectionTypeEmbeddingAttentionBias(
                model_dim=model_dimension,
                num_heads=number_of_heads,
                num_connection_types=self.shortest_path_length_upperbound+5,
            ) if 'shortest_path_length' in self.attention_biases else None
            self.attention_bias_edge_complementary_values = DiscreteConnectionTypeEmbeddingAttentionBiasComplementaryValues(
                model_dim=model_dimension,
                num_heads=number_of_heads,
                num_connection_types=PCQM4MV2_METADATA.num_discrete_bond_types + 4,
            ) if 'edge' in self.attention_biases else None
            self.attention_bias_shortest_path_length_complementary_values = DiscreteConnectionTypeEmbeddingAttentionBiasComplementaryValues(
                model_dim=model_dimension,
                num_heads=number_of_heads,
                num_connection_types=self.shortest_path_length_upperbound + 5,
            ) if 'shortest_path_length' in self.attention_biases else None
            self.attention_bias_shortest_path = PathTrajectoryEncodingAttentionBias(
                num_heads=number_of_heads,
                code_dim=self.path_encoding_code_dim,
                maximum_supported_path_length=path_encoding_length_upperbound
            ) if 'shortest_path' in self.attention_biases else None
        else:
            self.attention_bias_edge = torch.nn.ModuleDict({i: DiscreteConnectionTypeEmbeddingAttentionBias(
                model_dim=model_dimension,
                num_heads=number_of_heads,
                num_connection_types=PCQM4MV2_METADATA.num_discrete_bond_types + 4,
            ) if 'edge' in self.attention_biases else None for i in range(self.number_of_layers)})
            self.attention_bias_edge_complementary_values = torch.nn.ModuleDict({i: DiscreteConnectionTypeEmbeddingAttentionBiasComplementaryValues(
                model_dim=model_dimension,
                num_heads=number_of_heads,
                num_connection_types=PCQM4MV2_METADATA.num_discrete_bond_types + 4,
            ) if 'edge' in self.attention_biases else None for i in range(self.number_of_layers)})
            self.attention_bias_shortest_path_length_complementary_values = torch.nn.ModuleDict({i: DiscreteConnectionTypeEmbeddingAttentionBiasComplementaryValues(
                model_dim=model_dimension,
                num_heads=number_of_heads,
                num_connection_types=self.shortest_path_length_upperbound + 5,
            ) if 'shortest_path_length' in self.attention_biases else None for i in range(self.number_of_layers)})
            self.attention_bias_shortest_path_length = torch.nn.ModuleDict({i: DiscreteConnectionTypeEmbeddingAttentionBias(
                model_dim=model_dimension,
                num_heads=number_of_heads,
                num_connection_types=self.shortest_path_length_upperbound+5,
            ) if 'shortest_path_length' in self.attention_biases else None for i in range(self.number_of_layers)})
            self.attention_bias_shortest_path = torch.nn.ModuleDict({i: PathTrajectoryEncodingAttentionBias(
                num_heads=number_of_heads,
                code_dim=self.path_encoding_code_dim,
                maximum_supported_path_length=path_encoding_length_upperbound
            ) if 'shortest_path' in self.attention_biases else None for i in range(self.number_of_layers)})

        self.layered_stem = torch.nn.ModuleList([
            EncoderLayer(
                model_dim=model_dimension,
                feedforward_dim=feedforward_dimension,
                dropout_rate=dropout,
                attention_dropout_rate=attention_dropout,
                num_heads=number_of_heads,
            ) for _ in range(self.number_of_layers)
        ])

        self.toeplitz = toeplitz
        if self.toeplitz:
            self.toeplitz_row = torch.nn.Parameter(torch.ones(number_of_heads, 61))
            self.toeplitz_col = torch.nn.Parameter(torch.ones(number_of_heads, 61))

    def perturb_node_representations(self, node_features):
        if self.training and self.perturbation > 0:
            perturbation = torch.empty_like(node_features).uniform_(
                -self.perturbation, self.perturbation
            )
            node_features = node_features + perturbation
        return node_features

    def forward(
            self,
            batch: Dict[str, Any],
            return_node_reps: bool = False
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
        # batched_graph = batch['graphs']

        # - getting the encoded node types (with the offset applied)
        node_type = batch['node_type']

        # - connection types (edge types tokenized + no edge, self edge, and task edge
        node2node_connection_types = batch['node2node_connection_type']
        shortest_path_length_types = batch['node2node_shortest_path_length_type']

        if 'shortest_path_feature_trajectory' in batch:
            shortest_path_feature_trajectory = batch['shortest_path_feature_trajectory']
        else:
            shortest_path_feature_trajectory = None

        device = node_type.device

        # - performing the representation step for nodes
        node_features = self.node_encoder(node_type)

        # - node degree centrality
        if self.encode_node_degree_centrality:
            node_features += self.node_degree_centrality_encoder(1+batch['node_degree_centrality'].clip(max=self.node_degree_upperbound+1))

        # - perturbation, if requested
        node_features = self.perturb_node_representations(node_features=node_features)

        # - getting the node counts (including the task node)
        graph_node_counts = batch['node_counts']
        max_node_count = graph_node_counts.max().item()
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
                    shortest_path_feature_trajectory=shortest_path_feature_trajectory,
                    edge_bias_guide=self.attention_bias_edge[i],
                    shortest_path_length_bias_guide=self.attention_bias_shortest_path_length[i],
                    shortest_path_bias_guide=self.attention_bias_shortest_path[i],
                    edge_bias_guide_complementary_values=self.attention_bias_edge_complementary_values[i],
                    shortest_path_length_bias_guide_complementary_values=self.attention_bias_shortest_path_length_complementary_values[i],
                    mask=mask,
                    toeplitz=(self.toeplitz_row, self.toeplitz_col) if self.toeplitz else None
                )
            else:
                node_features = encoder_layer(
                    node_features,
                    distance=shortest_path_length_types,
                    connection_reps=node2node_connection_types,
                    shortest_path_feature_trajectory=shortest_path_feature_trajectory,
                    edge_bias_guide=self.attention_bias_edge,
                    shortest_path_length_bias_guide=self.attention_bias_shortest_path_length,
                    shortest_path_bias_guide=self.attention_bias_shortest_path,
                    edge_bias_guide_complementary_values=self.attention_bias_edge_complementary_values,
                    shortest_path_length_bias_guide_complementary_values=self.attention_bias_shortest_path_length_complementary_values,
                    mask=mask,
                    toeplitz=(self.toeplitz_row, self.toeplitz_col) if self.toeplitz else None
                )

        # - gathering the representations for the task node as graph embeddings
        last_node_indices = graph_node_counts - 1
        graph_reps = torch.gather(node_features, 1,
                                  last_node_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.model_dim)).squeeze(1)

        if return_node_reps:
            return graph_reps, node_features
        else:
            return graph_reps
