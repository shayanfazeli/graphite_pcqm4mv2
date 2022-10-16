from typing import List, Dict, Any
import torch
from torch_geometric.data import Data, Batch


def get_device(device: int) -> torch.device:
    """
    Returns a `torch.device` object.

    Parameters
    ----------
    device: `int`
        The device to use. -1 means cpu.
    """
    if device == -1:
        return torch.device("cpu")
    else:
        return torch.device("cuda:" + str(device))


def move_batch_to_device(batch, device):
    if isinstance(batch, Batch):
        batch = batch.to(device)
    elif isinstance(batch, Dict):
        for k in batch:
            if isinstance(batch[k], torch.Tensor) or isinstance(batch[k], Batch) or isinstance(batch[k], Data):
                batch[k] = batch[k].to(device)
            elif isinstance(batch[k], List):
                batch[k] = [move_batch_to_device(e) for e in batch[k]]
            elif isinstance(batch[k], Dict):
                batch[k] = {k2: move_batch_to_device(v) for k2, v in batch[k].items()}
            else:
                raise Exception(f"unsupported batch element")
    else:
        raise Exception(f"unrecognized batch type: {type(batch)}")

    return batch
