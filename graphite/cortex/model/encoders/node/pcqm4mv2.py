import torch
import torch.nn


class EmbedPCQM4Mv2NodeFeatures(torch.nn.Module):
    def __init__(self, model_dim: int, codebook_length: int = 334, padding_idx=None):
        super(EmbedPCQM4Mv2NodeFeatures, self).__init__()
        self.codebook = torch.nn.Embedding(codebook_length, model_dim, padding_idx=padding_idx)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.codebook.weight[node_features.long()].float().sum(dim=-2)
