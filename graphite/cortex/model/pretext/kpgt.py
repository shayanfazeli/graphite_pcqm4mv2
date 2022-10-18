# todo: complete

from typing import Dict, List, Any, Tuple
from .base import BasePretextModule
import torch
import torch.nn
import torch.linalg
import pytorch_metric_learning.distances as distance_lib
from torch_geometric.data import Data, Batch
from graphite.data.utilities.sequence_collate import get_pad_sequence_from_batched_reps, pad_sequence_2d
from .. import CustomMLPHead

class KPGTPretext(BasePretextModule):
    def __init__(self, *args, **kwargs):
        super(KPGTPretext, self).__init__(*args, **kwargs)

    def prepare_pretext_inputs(
            self,
            batch,
            graph_reps,
            node_reps,
            outputs
    ):
        pass

    def compute_pretext_loss(self, *args, **kwargs):
        pass

    def update_outputs(self, *args, **kwargs):
        pass