from typing import Tuple, Dict, List
import torch


def get_padding_shape(orig_shape: Tuple[int], pad_dim: int, max_len: int) -> Tuple[int]:
    output_shape = list(orig_shape)
    output_shape[pad_dim] = max_len - output_shape[pad_dim]
    return tuple(output_shape)


def pad_sequence(elements: List[torch.Tensor], pad_dim: int = 0, stack_dim: int = 0, max_len: int = None, pad_value=0):
    # - getting the device
    device = elements[0].device

    if max_len is None:
        max_len = max([e.shape[pad_dim] for e in elements])

    output = torch.stack([torch.cat(
        [e, pad_value + torch.zeros(get_padding_shape(e.shape, pad_dim=pad_dim, max_len=max_len)).to(device)],
        dim=pad_dim) for e in elements], dim=stack_dim)
    seq_len = torch.tensor([e.shape[pad_dim] for e in elements]).long().to(device)
    return output, seq_len


def pad_sequence_2d(elements: List[torch.Tensor], pad_dims: Tuple[int, int] = (0, 1), stack_dim: int = 0,
                    max_len: int = None, pad_value=0):
    # - getting the device
    device = elements[0].device

    if max_len is None:
        max_len_1 = max([e.shape[pad_dims[0]] for e in elements])
        max_len_2 = max([e.shape[pad_dims[1]] for e in elements])
        max_len = max(max_len_1, max_len_2)

    output = [torch.cat(
        [e, pad_value + torch.zeros(get_padding_shape(e.shape, pad_dim=pad_dims[0], max_len=max_len)).to(device)],
        dim=pad_dims[0]) for e in elements]

    output = torch.stack([torch.cat(
        [e, pad_value + torch.zeros(get_padding_shape(e.shape, pad_dim=pad_dims[1], max_len=max_len)).to(device)],
        dim=pad_dims[1]) for e in output], dim=stack_dim)

    seq_len = torch.tensor([e.shape[pad_dims[0]] for e in elements]).long().to(device)
    return output, seq_len
