import argparse
import wandb
import wandb.util

from graphite.utilities.logging.logger import get_logger

logger = get_logger(__name__)


def initialize_wandb(args: argparse.Namespace) -> None:
    if args.id is None:
        assert not args.resume, "you have chosen to resume a training process, therefore," \
                                "please provide the corresponding wandb to resume logging."
        id = wandb.util.generate_id()
        args.id = id
        wandb.init(id=args.id)
        logger.info("`wandb` session is initialized and ready to START the operation.")
    else:
        assert args.resume, "you have provided the id, so you must be resuming an experiment, however, the resume flag is not provided"
        wandb.init(id=args.id, resume="must")
        logger.info("`wandb` session is initialized and ready to RESUME the operation.")
