from typing import Dict, Any, List
import torch
import torch.nn
import graphite.cortex.model as model_lib
import graphite.cortex.optimization.loss as loss_lib


# todo: validate this and complete it
class SingleModelSingleViewPretrainingWithPretexts(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_layers: int,
            model_config: Dict[str, Any],
            loss_config: Dict[str, Any],
            pretext_configs: Dict[str, Dict[str, Any]]
    ):
        super(SingleModelSingleViewPretrainingWithPretexts, self).__init__()

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

        self.pretext_heads = torch.nn.ModuleDict(
            {k: getattr(model_lib, e['type'])(**e['args']) for k, e in pretext_configs.items()}
        )

    def forward(self, batch):
        y = batch['y']
        latent_reps, node_reps = self.model(batch, return_node_reps=True)

        # - core regression_loss
        preds = self.projector(latent_reps).squeeze()
        loss = self.criterion(preds, y)

        outputs = dict(
            preds=preds,
            y=y,
            loss=loss
        )

        for pretext in self.pretext_heads:
            pretext_loss, outputs = self.pretext_heads[pretext](
                graph_reps=latent_reps,
                node_reps=node_reps,
                outputs=outputs,
            )
            loss = loss + pretext_loss

        return loss, outputs
