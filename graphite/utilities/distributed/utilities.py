import os
import argparse
import builtins
import torch
import torch.cuda
import torch.distributed
import apex
from apex.parallel.LARC import LARC


def setup_distributed_training_if_requested(args) -> argparse.Namespace:
    # inspired by `https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904`
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        # suppress printing if not on master gpu, only in the case of distributed.
        # note that the default value for rank is -1
        if args.rank != 0:
            def print_pass(*args):
                pass
            builtins.print = print_pass

    return args


def prepare_model_for_ddp_if_requested(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    assert args.gpu is not None, f"`args.gpu` cannot be `None` in the distributed processes."
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    return model


def sync_batchnorms(model: torch.nn.Module, args: argparse.Namespace, strategy: str) -> torch.nn.Module:
    if args.sync_bn == "pytorch":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif args.sync_bn == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.world_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    else:
        raise ValueError(f"unknown strategy for batchnorm synchronization: {strategy}")
    return model


def larc_optimizer(optimizer, trust_coefficient: float = 0.001, clip=False):
    optimizer = LARC(optimizer=optimizer, trust_coefficient=trust_coefficient, clip=clip)
    return optimizer


def get_non_ddp_model_from_ddp_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.module
