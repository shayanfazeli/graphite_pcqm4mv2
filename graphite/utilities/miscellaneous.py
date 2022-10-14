import torch
import torch.nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def toeplitz(c):
    # - toeplitz matrix
    vals = torch.cat((c, c[1:].flip(0)))
    shape = len(c), len(c)
    i, j = torch.ones(*shape).nonzero().T
    return vals[j-i].reshape(*shape)
