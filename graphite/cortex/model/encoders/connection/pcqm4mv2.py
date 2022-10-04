import torch
import torch.nn


class EmbedPCQM4Mv2EdgeType(torch.nn.Module):
    def __init__(self, model_dim: int, codebook_length: int = 31):
        super(EmbedPCQM4Mv2EdgeType, self).__init__()

        self.codebook = torch.nn.Embedding(codebook_length, model_dim)

    def forward(self, node2node_connection_types: torch.Tensor) -> torch.Tensor:
        return self.codebook.weight[node2node_connection_types.long()].float().sum(dim=-2)


class EmbedPCQM4Mv2ShortestPathLengthType(torch.nn.Module):
    def __init__(self, model_dim: int, codebook_length: int = 260):
        super(EmbedPCQM4Mv2ShortestPathLengthType, self).__init__()

        self.codebook = torch.nn.Embedding(codebook_length, model_dim)

    def forward(self, node2node_shortest_path_length_type: torch.Tensor) -> torch.Tensor:
        return self.codebook.weight[node2node_shortest_path_length_type.long()].float().sum(dim=-2)
