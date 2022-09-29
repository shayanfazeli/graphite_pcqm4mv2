import torch
import torch.nn
import graphite.cortex.model as model_lib


class Regressor(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_layers: int,
            model_config
    ):
        super(Regressor, self).__init__()

        # - core model
        self.model = getattr(model_lib, model_config['type'])(**model_config['args'])

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
        latent_reps = self.model(batch_data)
        preds = self.projector(latent_reps).squeeze()

        return dict(
            latent_reps=latent_reps,
            preds=preds
        )