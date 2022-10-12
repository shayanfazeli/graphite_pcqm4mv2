import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, ConstantLR, ExponentialLR, SequentialLR


def param(model, lr: float, wd: float):
    param_groups = [{'params': [], 'lr': lr,   'weight_decay': 0},
                    {'params': [], 'lr': lr,   'weight_decay': wd},
                    {'params': [], 'lr': lr/2, 'weight_decay': wd*2}]

    for n, p in model.named_parameters():
        if n.find('_embedding_') > 0:
            param_groups[0]['params'].append(p)
        elif n.find('head') > 0 or n.find('projector') > 0:
            param_groups[2]['params'].append(p)
        elif n.endswith('scale'):
            param_groups[0]['params'].append(p)
        elif n.endswith('bias'):
            param_groups[0]['params'].append(p)
        elif n.endswith('weight'):
            param_groups[1]['params'].append(p)
        else:
            raise Exception('Unknown parameter name:', n)

    return param_groups


def coatgin_optim_and_scheduler(
        model,
        lr: float,
        wd: float,
        warmups: int
):
    optimizer = optim.AdamW(param(model, lr, wd), lr=lr, weight_decay=wd, betas=(0.9, 0.998))
    sched0 = LinearLR(optimizer, 1e-4, 1.0)
    sched1 = ConstantLR(optimizer, 1.0)
    sched2 = ExponentialLR(optimizer, 0.01**0.01)
    scheduler = SequentialLR(optimizer, [sched0, sched1, sched2], [warmups//5, warmups])
    scheduling_interval = 'epoch'
    return optimizer, scheduler, scheduling_interval
