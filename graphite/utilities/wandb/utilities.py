from typing import Dict, Any
import argparse
import os
import wandb
import wandb.util
import torch.distributed
from graphite.utilities.io.files_and_folders import clean_folder
from graphite.utilities.logging.logger import get_logger, log_message

logger = get_logger(__name__)


def initialize_wandb(args: argparse.Namespace, config: Dict[str, Any] = None) -> None:
    """
    Initializing the wandb session

    Parameters
    ----------
    args: `argparse.Namespace`, required
        Arguments that are parsed from `graphite_train` script.

    config: `Dict[str, Any]`, optional (default=None)
        The parsed configuration
    """
    if not (args.distributed and args.local_rank != 0):
        if args.id is None:
            assert not args.resume, "you have chosen to resume a training process, therefore," \
                                    "please provide the corresponding wandb to resume logging."
            id = wandb.util.generate_id()
            args.id = id
            args.logdir = os.path.join(args.logdir, args.project, args.name, f'seed_{args.seed}')

            if args.clean:
                clean_folder(args.logdir)
            os.makedirs(os.path.join(args.logdir, 'ckpts'), exist_ok=True)

            wandb.init(id=args.id, project=args.project, name=args.name, config=config, dir=args.logdir)
            log_message(logger, "`wandb` session is initialized and ready to START the operation.", args)
        else:
            assert args.resume, "you have provided the id, so you must be resuming an experiment, however, the resume flag is not provided"
            args.logdir = os.path.join(args.logdir, args.project, args.name, f'seed_{args.seed}')
            wandb.init(id=args.id, project=args.project, name=args.name, resume="must", dir=args.logdir)
            log_message(logger, "`wandb` session is initialized and ready to RESUME the operation.", args)

    if args.distributed:
        torch.distributed.barrier()