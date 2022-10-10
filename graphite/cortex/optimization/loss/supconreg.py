import torch
import torch.nn
import pytorch_metric_learning.distances as pml_distances


class SupConRegLoss(torch.nn.Module):
    def __init__(
            self,
            temperature: float = 0.07,
            label_distance_fn: torch.nn.Module = pml_distances.LpDistance(normalize_embeddings=False, p=1, power=1, is_inverted=False),
            features_similarity_fn: torch.nn.Module = pml_distances.DotProductSimilarity(),
            reduction: str = 'mean'
    ):
        super(SupConRegLoss, self).__init__()
        self.temperature = temperature
        self.label_distance_fn = label_distance_fn
        self.features_similarity_fn = features_similarity_fn
        assert reduction in ['mean', 'sum', 'none']
        self.reduction = reduction


    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        assert labels.ndim == 1
        assert features.ndim == 2
        n = features.shape[0]

        # - step 1: computing the similarities between features
        sims = torch.exp(torch.div(self.features_similarity_fn(features), self.temperature))  # dim: n, n

        # - step 2: computing label distances
        label_distances = self.label_distance_fn(labels.unsqueeze(-1))  # dim: n, n

        # - step 3: broadcasting
        ijk_mat = label_distances.unsqueeze(1).repeat(1, n, 1) # dim: n, n, n

        # - note that in the above, if you take ijk_mat[i,j, :], you will have the layout of differences
        # between all items and i.

        # - step 3: creating the negative-set mask for each i,j
        if sims.dtype == torch.float16:
            mask = (ijk_mat > label_distances.unsqueeze(-1)).half()
        else:
            mask = (ijk_mat > label_distances.unsqueeze(-1)).float()  # dim: n, n, n
        # - in the above, if you take mask[i,j], you will get the dim `n` mask of examples that are
        # negative sets

        # - computing the denominator
        denominator = torch.einsum('ijk,ik->ij', mask, sims)  # dim: n, n
        # - in the above, denominator[i, j] corresponds to the sum in the denominator of the formula (1)
        # in `https://arxiv.org/pdf/2210.01189.pdf`

        # - step 4: creating farthest example mask to ignore:
        ignore_mask = (denominator != 0)
        # - updating similarities with it:
        sims = sims * ignore_mask

        # - removing selves
        sims = sims.fill_diagonal_(0)

        # - step 5: computing all l_ij losses:
        losses = torch.div(sims, denominator.clip(min=1e-4))  # of dim n, n

        # - step 5: computing all anchor losses
        losses = losses.sum(dim=1) / torch.tensor(n-1)

        if self.reduction == 'none':
            return losses
        elif self.reduction == 'mean':
            return torch.mean(losses)
        elif self.reduction == 'sum':
            return torch.sum(losses)
        else:
            raise NotImplementedError()



