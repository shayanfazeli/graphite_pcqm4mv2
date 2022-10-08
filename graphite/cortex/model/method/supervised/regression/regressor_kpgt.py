from typing import Dict, Any, List
import torch
import torch.nn
import graphite.cortex.model as model_lib
import graphite.cortex.optimization.loss as loss_lib


class RegressorWithKPGTRegularization(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_layers: int,
            model_config: Dict[str, Any],
            loss_config: Dict[str, Any],
            kpgt_loss_config: Dict[str, Any],
    ):
        super(RegressorWithKPGTRegularization, self).__init__()

        # - core model
        self.add_module('model', getattr(model_lib, model_config['type'])(**model_config['args']))

        self.criterion = getattr(loss_lib, loss_config['type'])(**loss_config['args'])

        # - preparing the criterions for kpgt regularization task
        self.kpgt_fingerprint_loss_coeff = kpgt_loss_config['fingerprint']['factor']
        self.kpgt_descriptor_loss_coeff = kpgt_loss_config['descriptor']['factor']
        self.criterion_fingerprint = getattr(loss_lib, kpgt_loss_config['fingerprint']['type'])(**kpgt_loss_config['fingerprint']['args'])
        self.criterion_descriptor = getattr(loss_lib, kpgt_loss_config['descriptor']['type'])(**kpgt_loss_config['descriptor']['args'])

        # - projector
        self.projector = [torch.nn.LayerNorm(input_dim)]

        for _ in range(num_layers):
            self.projector += [torch.nn.Sequential(
                torch.nn.Linear(input_dim, input_dim),
                torch.nn.GELU()
            )]

        self.projector += [torch.nn.Linear(input_dim, output_dim)]
        self.projector = torch.nn.Sequential(*self.projector)

        self.kpgt_fingerprint_head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.LayerNorm(input_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(input_dim, 512),
        )

        self.kpgt_descriptor_head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.LayerNorm(input_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(input_dim, 200),
        )

    def forward(self, batch_data):
        # - getting the latent representation and main loss
        y = batch_data['y']
        latent_reps = self.model(batch_data)
        preds = self.projector(latent_reps).squeeze()
        loss = self.criterion(preds, y)

        # - kpgt regularization head
        kpgt_logits_fp = self.kpgt_fingerprint_head(latent_reps)
        kpgt_logits_desc = self.kpgt_descriptor_head(latent_reps)
        kpgt_loss_fp = self.criterion_fingerprint(kpgt_logits_fp, batch_data['molecule_fingerprint'])
        kpgt_loss_desc = self.criterion_descriptor(kpgt_logits_desc, batch_data['molecule_descriptor'])
        loss_kpgt = self.kpgt_fingerprint_loss_coeff * kpgt_loss_fp + self.kpgt_descriptor_loss_coeff * kpgt_loss_desc

        return loss + loss_kpgt, dict(
            latent_reps=latent_reps,
            preds=preds,
            y=y,
            loss=loss,
            loss_kpgt_fp=kpgt_loss_fp,
            loss_kpgt_desc=kpgt_loss_desc,
            loss_kpgt=loss_kpgt
        )
