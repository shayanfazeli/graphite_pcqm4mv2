import torch
import torch.nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# from `https://stackoverflow.com/questions/69809789/is-there-any-way-to-create-a-tensor-with-a-specific-pattern-in-pytorch`
def toeplitz(c, r):
    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T
    return vals[j-i].reshape(*shape)
