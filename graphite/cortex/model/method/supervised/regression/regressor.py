from typing import Dict, Any, List
import torch
import torch.nn
import graphite.cortex.model as model_lib
import graphite.cortex.optimization.loss as loss_lib


class Regressor(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_layers: int,
            model_config: Dict[str, Any],
            loss_config: Dict[str, Any]
    ):
        super(Regressor, self).__init__()

        # - core model
        self.add_module('model', getattr(model_lib, model_config['type'])(**model_config['args']))

        self.criterion = getattr(loss_lib, loss_config['type'])(**loss_config['args'])

        # - projector
        self.projector = [torch.nn.LayerNorm(input_dim)]

        for _ in range(num_layers):
            self.projector += [torch.nn.Sequential(
                torch.nn.Linear(input_dim, input_dim),
                torch.nn.GELU()
            )]

        self.projector += [torch.nn.Linear(input_dim, output_dim)]
        self.projector = torch.nn.Sequential(*self.projector)

    def forward(self, batch_data):
        y=batch_data['y']
        latent_reps = self.model(batch_data)
        preds = self.projector(latent_reps).squeeze()
        loss = self.criterion(preds, y)

        return loss, dict(
            latent_reps=latent_reps,
            preds=preds,
            y=y
        )
