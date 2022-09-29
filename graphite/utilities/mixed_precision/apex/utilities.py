import apex
import torch


def apex_initialize_optimizer(model: torch.nn.Module, optimizer: torch.optim.Optimizer, opt_level: str = "O1"):
    model, optimizer = apex.amp.initialize(model, optimizer, opt_level=opt_level)
    return model, optimizer
