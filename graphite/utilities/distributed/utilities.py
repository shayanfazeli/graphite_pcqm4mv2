import os
import argparse
import builtins
import torch
import torch.cuda
import torch.distributed
import apex
from apex.parallel.LARC import LARC


def setup_distributed_training_if_requested(args: argparse.Namespace) -> argparse.Namespace:
    """
    Setting up the distributed learning, support for
    `torch.distributed.launch`, `SLURM`, and `torchrun`.

    Parameters
    ----------
    args: `argparse.Namespace`, required
        The arguments

    Returns
    ----------
    `argparse.Namespace`:
        The modified argument namespace to be used by the script.
    """
    # - first checking the os environmental variables
    # for presence of distributed guidelines
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = True

    # - checking slurm job
    args.is_slurm_job = "SLURM_JOB_ID" in os.environ
    if args.is_slurm_job:
        args.distributed = True

    # - the distributed
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif args.is_slurm_job:  # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
        else:
            args.local_rank = int(os.environ['LOCAL_RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.rank = int(os.environ['RANK'])
            args.dist_url = str(os.environ['MASTER_ADDR'])
            args.gpu = args.local_rank

        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

        # suppress printing if not on master gpu, only in the case of distributed.
        # note that the default value for rank is -1
        if args.rank != 0:
            def print_pass(*args):
                pass
            builtins.print = print_pass

    return args


def prepare_model_for_ddp_if_requested(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    """
    Parameters
    ----------
    args: `argparse.Namespace`, required
        The arguments
    Returns
    ----------
    """
    assert args.gpu is not None, f"`args.gpu` cannot be `None` in the distributed processes."
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    return model


def sync_batchnorms(model: torch.nn.Module, args: argparse.Namespace, strategy: str) -> torch.nn.Module:
    """
    Parameters
    ----------
    model: `torch.nn.Module`, required
        The model

    args: `argparse.Namespace`, required
        The arguments

    strategy: `str`, required
        The strategy for batchnorm synchronization from `pytorch` or `apex`.
        If `None` is passed, nothing would be done.

    Returns
    ----------
    `torch.nn.Module`: the modified module.
    """
    if strategy is None:
        return model
    elif strategy == "pytorch":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    elif strategy == "apex":
        # with apex syncbn we sync bn per group because it speeds up computation
        # compared to global syncbn
        process_group = apex.parallel.create_syncbn_process_group(args.world_size)
        model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
    else:
        raise ValueError(f"unknown strategy for batchnorm synchronization: {strategy}")
    return model


def larc_optimizer(
        optimizer: torch.optim.Optimizer,
        trust_coefficient: float = 0.001,
        clip=False):
    """
    Parameters
    ----------
    optimizer: `torch.optim.Optimizer`, required
        The optimizer

    Returns
    ----------
    """
    optimizer = LARC(optimizer=optimizer, trust_coefficient=trust_coefficient, clip=clip)
    return optimizer


def get_non_ddp_model_from_ddp_module(model: torch.nn.Module) -> torch.nn.Module:
    """
    Parameters
    ----------
    model: `torch.nn.Module`, required
        The model

    Returns
    ----------
    `torch.nn.Module`:
        The module
    """
    return model.module
