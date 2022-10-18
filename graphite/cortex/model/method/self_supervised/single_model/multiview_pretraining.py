from typing import Dict, Any, List
import torch
import torch.nn
import graphite.cortex.model as model_lib
import graphite.cortex.optimization.loss as loss_lib


# todo: validate this and complete it
class SingleModelMultiViewPretrainingWithPretexts(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_layers: int,
            model_config: Dict[str, Any],
            loss_config: Dict[str, Any],
            pretext_configs: Dict[str, Dict[str, Any]]
    ):
        super(SingleModelMultiViewPretrainingWithPretexts, self).__init__()
        # raise Exception("not validated yet.")
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

    def forward(self, batch_list):
        y = batch_list[0]['y']
        latent_reps_list = list()
        node_reps_list = list()
        for i, batch_data in enumerate(batch_list):
            latent_reps, node_reps = self.model(batch_data, return_node_reps=True)
            latent_reps_list.append(latent_reps)
            node_reps_list.append(node_reps)

        # - core regression_loss
        loss = 0
        preds = 0
        for i, latent_reps in enumerate(latent_reps_list):
            preds_tmp = self.projector(latent_reps).squeeze()
            loss = loss + self.criterion(preds_tmp, y)
            preds = preds + preds_tmp

        preds = preds / float(len(latent_reps_list))
        loss = loss / float(len(latent_reps_list))
        outputs = dict(
            preds=preds,
            y=y,
            loss=loss
        )

        for pretext in self.pretext_heads:
            pretext_loss, outputs = self.pretext_heads[pretext](
                graph_reps=latent_reps_list,
                node_reps=node_reps_list,
                outputs=outputs,
                batch=batch_list
            )
            loss = loss + pretext_loss

        return loss, outputs
