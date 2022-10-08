import abc

import torch
import torch.nn


class AttentionBiasBase(torch.nn.Module):
    def __init__(self):
        super(AttentionBiasBase, self).__init__()