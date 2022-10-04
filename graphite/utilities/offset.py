import torch


@torch.jit.script
def add_feature_position_offset(x, offset: int):
    assert x.ndim <= 2
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x
