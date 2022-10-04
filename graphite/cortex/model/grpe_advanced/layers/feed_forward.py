import torch
import torch.nn


class FeedForwardNetwork(torch.nn.Module):
    """
    A simple feed forward module which will take the input to a projected dimension
    and back to the original one.

    Parameters
    ----------
    input_dim: `int`, required
        The input size

    auxiliary_dim: `int`, required
        The auxiliary dimension for the projection

    dropout_rate: `float`, required
        The dropout rate
    """
    def __init__(
            self,
            input_dim: int,
            auxiliary_dim: int,
            dropout_rate: float
    ):
        """constructor"""
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, auxiliary_dim)
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer2 = torch.nn.Linear(auxiliary_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: `torch.Tensor`, required
            The input tensor of dim `..., input_dim`

        Returns
        ----------
        `torch.Tensor`:
            The output tensor of dim `..., input_dim`
        """
        x = self.layer1(x)
        x = self.gelu(x)
        # x = self.dropout(x)
        x = self.layer2(x)
        return x
