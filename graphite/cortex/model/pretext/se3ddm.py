from typing import Dict, List, Any, Tuple
from .base import BasePretextModule
import torch
import torch.nn
import torch.linalg
import pytorch_metric_learning.distances as distance_lib
from torch_geometric.data import Data, Batch
from graphite.data.utilities.sequence_collate import get_pad_sequence_from_batched_reps, pad_sequence_2d
from .. import CustomMLPHead

# todo: complete and verify
# todo: refactor this
@torch.jit.script
def get_2d_masks_from_sequence_lengths(sequence_lengths, max_len: int):
    output = []
    for l in sequence_lengths:
        m = torch.zeros((max_len, max_len))
        m[:l, :l] = 1
        output.append(m)
    return torch.stack(output, dim=0)


class SE3DDMPretext(BasePretextModule):
    """
    The implementation of the article "Molecular Geometry Pretraining with SE(3)-Invariant Denoising Distance Matching": [`https://arxiv.org/pdf/2206.13602.pdf`](https://arxiv.org/pdf/2206.13602.pdf)

    Parameters
    ----------
    scale: `float`, required
        The scale for the gaussian noise to be added to the pair-wise distances for perturbing them.

    mlp_distances_args: `Dict[str, Any]`, required
        Configuration for the MLP that works on the distances

    mlp_merge_args: `Dict[str, Any]`, required
        Configuration for the MLP that is used for fusion of node representations and the distance representations

    distance: `BaseDistance`, optional (default=LpDistance(p=2, power=1, normalize_embeddings=False))
        Distance module to compute pair-wise distances from 3D coordinates.
    """
    def __init__(
            self,
            scale: float,
            mlp_distances_args: Dict[str, Any],
            mlp_merge_args: Dict[str, Any],
            distance_config: Dict[str, Any] = dict(
                type='LpDistance',
                args=dict(
                    p=2,
                    power=1,
                    normalize_embeddings=False
                )
            ),
            beta: float = 1.0,
            mode: str = 'with_distances',
            *args,
            **kwargs):
        """constructor"""
        super(SE3DDMPretext, self).__init__(*args, **kwargs)
        self.scale = scale
        self.mlp_distances = CustomMLPHead(**mlp_distances_args)
        self.mlp_merge = CustomMLPHead(**mlp_merge_args)
        self.distance = getattr(distance_lib, distance_config['type'])(**distance_config['args'])
        self.mse = torch.nn.MSELoss()
        self.beta = beta
        self.mode = mode

    def prepare_pretext_inputs_with_positions(
            self,
            batch: List[Any],
            graph_reps: List[torch.Tensor],
            node_reps: List[torch.Tensor],
            outputs: Dict[str, Any]
    ):
        """preparing inputs"""
        assert len(batch) == len(graph_reps) == len(node_reps) == 2, "SE(3) DDM Pretext is only supported for 2-view training."

        inputs_dict = dict()
        if isinstance(batch[0], Batch):
            for i in [0, 1]:
                inputs_dict[f'node_reps_{i+1}'] = get_pad_sequence_from_batched_reps(
                    reps=node_reps[i],
                    batch_ptr=batch[i].ptr
                )
                inputs_dict[f'positions_3d_{i+1}'] = get_pad_sequence_from_batched_reps(
                    reps=batch[i].positions_3d,
                    batch_ptr=batch[i].ptr
                )
            inputs_dict['node_counts'] = batch[0].node_counts
        elif isinstance(batch[0], Dict):
            for i in [0, 1]:
                inputs_dict[f'node_reps_{i+1}'] = node_reps[i]
                inputs_dict[f'positions_3d_{i+1}'] = batch[i]['positions_3d']
            inputs_dict['node_counts'] = batch[0]['node_counts'] - 1
        else:
            raise Exception(f"Unknown batch type: {type(batch)}")

        return inputs_dict

    def prepare_pretext_inputs_with_distances(
            self,
            batch: List[Any],
            graph_reps: List[torch.Tensor],
            node_reps: List[torch.Tensor],
            outputs: Dict[str, Any]
    ):
        """preparing inputs"""
        assert len(batch) == len(graph_reps) == len(node_reps) == 2, "SE(3) DDM Pretext is only supported for 2-view training."

        inputs_dict = dict()
        if isinstance(batch[0], Batch):
            for i in [0, 1]:
                inputs_dict[f'node_reps_{i+1}'] = get_pad_sequence_from_batched_reps(
                    reps=node_reps[i],
                    batch_ptr=batch[i].ptr
                )
                inputs_dict[f'd{i+1}'] = batch[i].pairwise_distances
            inputs_dict['node_counts'] = batch[0].node_counts
        elif isinstance(batch[0], Dict):
            for i in [0, 1]:
                inputs_dict[f'node_reps_{i+1}'] = node_reps[i]
                inputs_dict[f'd{i+1}'] = batch[i]['pairwise_distances']
            inputs_dict['node_counts'] = batch[0]['node_counts'] - 1
        else:
            raise Exception(f"Unknown batch type: {type(batch)}")

        return inputs_dict

    def prepare_pretext_inputs(
            self,
            *args,
            **kwargs
    ):
        if self.mode == 'with_distances':
            return self.prepare_pretext_inputs_with_distances(*args, **kwargs)
        elif self.mode == 'with_positions':
            return self.prepare_pretext_inputs_with_positions(*args, **kwargs)
        else:
            raise Exception(f"Unknown mode for se3ddm pretext: mode={self.mode}")

    def compute_pretext_loss_with_positions(
            self,
            node_reps_1: torch.Tensor,
            positions_3d_1: torch.Tensor,
            node_reps_2: torch.Tensor,
            positions_3d_2: torch.Tensor,
            node_counts: torch.LongTensor
    ):
        """
        Parameters
        ----------
        node_reps_1: `torch.Tensor`, required
            The computed latent node representations of `dim=(batch_size, max_num_node, model_dim)`

        positions_3d_1: `torch.Tensor`, required
            The padded node positions of `dim=(batch_size, max_num_node, 3)`

        node_reps_2: `torch.Tensor`, required
            The computed latent node representations of `dim=(batch_size, max_num_node, model_dim)`

        positions_3d_2: `torch.Tensor`, required
            The padded node positions of `dim=(batch_size, max_num_node, 3)`

        node_counts: `torch.LongTensor`, required
            The node counts for each molecule in the batch

        Returns
        ----------
        `torch.Tensor`:
            Output scores of `dim=batch_size, max_num+node, max_num_node`
        """
        # - sanity checks
        assert isinstance(node_counts, torch.Tensor)
        assert node_counts.ndim == 1

        # - computing pair-wise distances
        d1 = self.get_pairwise_distances(positions_3d=positions_3d_1, node_counts=node_counts)
        d2 = self.get_pairwise_distances(positions_3d=positions_3d_2, node_counts=node_counts)

        d1_tilde = d1 + self.scale * torch.randn(d1.size(), device=d1.device)
        d2_tilde = d2 + self.scale * torch.randn(d2.size(), device=d2.device)

        # - computing scores
        s12 = self.score(pairwise_distance=d1_tilde, node_reps=node_reps_2)
        s21 = self.score(pairwise_distance=d2_tilde, node_reps=node_reps_1)

        # - computing the mask
        mask = get_2d_masks_from_sequence_lengths(node_counts, max_len=node_counts.max().item()).to(node_reps_1.device)
        term_1 = (self.scale ** self.beta) * ((s12 / self.scale) - ((d1 - d1_tilde) / (self.scale ** 2)))
        term_2 = (self.scale ** self.beta) * ((s21 / self.scale) - ((d2 - d2_tilde) / (self.scale ** 2)))

        term_1 = term_1 * mask
        term_2 = term_2 * mask

        loss = torch.linalg.norm(term_1.flatten(1), ord=2, dim=1) + torch.linalg.norm(term_2.flatten(1), ord=2, dim=1)
        loss = loss.mean()

        return loss

    def compute_pretext_loss_with_distances(
            self,
            node_reps_1: torch.Tensor,
            d1: torch.Tensor,
            node_reps_2: torch.Tensor,
            d2: torch.Tensor,
            node_counts: torch.LongTensor
    ):
        """
        Parameters
        ----------
        node_reps_1: `torch.Tensor`, required
            The computed latent node representations of `dim=(batch_size, max_num_node, model_dim)`

        d1: `torch.Tensor`, required
            The pair-wise distances `dim=(batch_size, max_num_node, max_num_node)`

        node_reps_2: `torch.Tensor`, required
            The computed latent node representations of `dim=(batch_size, max_num_node, model_dim)`

        d2: `torch.Tensor`, required
            The pair-wise distances `dim=(batch_size, max_num_node, max_num_node)`

        node_counts: `torch.LongTensor`, required
            The node counts for each molecule in the batch

        Returns
        ----------
        `torch.Tensor`:
            Output scores of `dim=batch_size, max_num+node, max_num_node`
        """
        # - sanity checks
        assert isinstance(node_counts, torch.Tensor)
        assert node_counts.ndim == 1

        d1_tilde = d1 + self.scale * torch.randn(d1.size(), device=d1.device)
        d2_tilde = d2 + self.scale * torch.randn(d2.size(), device=d2.device)

        # - computing scores
        s12 = self.score(pairwise_distance=d1_tilde, node_reps=node_reps_2)
        s21 = self.score(pairwise_distance=d2_tilde, node_reps=node_reps_1)

        # - computing the mask
        mask = get_2d_masks_from_sequence_lengths(node_counts, max_len=node_counts.max().item()).to(node_reps_1.device)
        term_1 = (self.scale ** self.beta) * ((s12 / self.scale) - ((d1 - d1_tilde) / (self.scale ** 2)))
        term_2 = (self.scale ** self.beta) * ((s21 / self.scale) - ((d2 - d2_tilde) / (self.scale ** 2)))

        term_1 = term_1 * mask
        term_2 = term_2 * mask

        loss = torch.linalg.norm(term_1.flatten(1), ord=2, dim=1) + torch.linalg.norm(term_2.flatten(1), ord=2, dim=1)
        loss = loss.mean()

        return loss

    def compute_pretext_loss(self, *args, **kwargs):
        if self.mode == 'with_distances':
            return self.compute_pretext_loss_with_distances(*args, **kwargs)
        elif self.mode == 'with_positions':
            return self.compute_pretext_loss_with_positions(*args, **kwargs)
        else:
            raise Exception("Unknown mode for se3ddm pretext")

    def update_outputs(self, outputs, loss, **kwargs):
        outputs['loss_se3ddm_pretext'] = loss.item()
        return outputs

    def score(
            self,
            pairwise_distance: torch.Tensor,
            node_reps: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pairwise_distance: `torch.Tensor`, required
            The pair-wise distances of `dim=(batch_size, max_num_node, max_num_node)`

        node_reps: `torch.Tensor`, required
            The computed latent node representations of `dim=(batch_size, max_num_node, model_dim)`

        Returns
        ----------
        `torch.Tensor`:
            Output scores of `dim=batch_size, max_num_node, max_num_node`
        """
        orig_size = pairwise_distance.size()  # batch_size, max_num_node, max_num_node

        # - projecting pair-wise distances to a latent emb
        z_d = self.mlp_distances(pairwise_distance.view(-1, 1)).reshape(*orig_size,
                                                                        -1)  # dim=batch_size, max_num_node, max_num_node, model_dim
        model_dim = z_d.size(-1)

        z_g_ij = node_reps.unsqueeze(1) + node_reps.unsqueeze(
            2)  # dim=batch_size, max_num_node, max_num_node, model_dim

        orig_size = list(z_g_ij.shape)[:-1]
        return self.mlp_merge((z_d + z_g_ij).view(-1, model_dim)).reshape(*orig_size)

    def get_pairwise_distances(self, positions_3d, node_counts) -> torch.Tensor:
        """
        Returns
        ----------
        `torch.Tensor`:
            The output of `dim=(batch_size, max_num_node, max_num_node)`, which contains
            the padded node-by-node distances per molecule.
        """
        return pad_sequence_2d(
            [self.distance(positions_3d[i, :n, :n]) for i, n in enumerate(node_counts)],
            max_len=node_counts.max(),
            pad_value=-1)[0]
