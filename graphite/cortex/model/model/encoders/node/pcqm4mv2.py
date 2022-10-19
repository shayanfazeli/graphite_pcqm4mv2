import torch
import torch.nn
import graphite.data.utilities.pcqm4mv2_meta as PCQM4MV2_META

class EmbedPCQM4Mv2NodeFeatures(torch.nn.Module):
    def __init__(self, model_dim: int, codebook_length: int = PCQM4MV2_META.task_node_feat+1, padding_idx=None):
        super(EmbedPCQM4Mv2NodeFeatures, self).__init__()
        self.codebook = torch.nn.Embedding(codebook_length, model_dim, padding_idx=padding_idx)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        return self.codebook.weight[node_features].sum(dim=-2)
