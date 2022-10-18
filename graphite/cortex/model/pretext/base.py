import abc
from typing import Union, List, Dict, Any
import torch
import torch.nn
from torch_geometric.data import Data, Batch


class BasePretextModule(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, weight: float):
        super(BasePretextModule, self).__init__()
        self.weight = weight

    @abc.abstractmethod
    def prepare_pretext_inputs(
            self,
            batch: Union[List[Any], Any],
            graph_reps: Union[List[torch.Tensor], torch.Tensor],
            node_reps: Union[List[torch.Tensor], torch.Tensor],
            outputs: Dict[str, Any]
    ):
        pass

    @abc.abstractmethod
    def compute_pretext_loss(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def update_outputs(self, *args, **kwargs):
        pass

    def forward(
            self,
            batch: Union[List[Any], Any],
            graph_reps: Union[List[torch.Tensor], torch.Tensor],
            node_reps: Union[List[torch.Tensor], torch.Tensor],
            outputs: Dict[str, Any]
    ):
        """
        Parameters
        ----------
        batch: `Union[List[Any], Any]`,
        graph_reps: `Union[List[torch.Tensor], torch.Tensor]`,
        node_reps: `Union[List[torch.Tensor], torch.Tensor]`,
        outputs: `Dict[str, Any]`, required
            Additional outputs bundle which will be updated via `update_outputs` method.
        """
        inputs = self.prepare_pretext_inputs(
            batch=batch,
            graph_reps=graph_reps,
            node_reps=node_reps,
            outputs=outputs
        )

        loss = self.compute_pretext_loss(**inputs)
        outputs = self.update_outputs(outputs=outputs, loss=loss, **inputs)
        return self.weight * loss, outputs
