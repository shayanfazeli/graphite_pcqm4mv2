import argparse
import logging


def get_logger(name, level=None):
    """
    Parameters
    ----------
    name: `str`, required
        Name of the logger.

    level: `str`, optional (default=None)
        Level of the logger. The default is None.

    Returns
    -------
    `logging.Logger`: the logger object will be returned.
    """
    level = logging.INFO if level is None else level
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def log_message(logger, msg, args: argparse.Namespace, level='info'):
    if 'distributed' not in args:
        getattr(logger, level)(msg)
    elif (not args.distributed) or (args.rank == 0):
        getattr(logger, level)(msg)


def grad_stats(logger, model, args):
    for n, p in model.named_parameters():
        if p.requires_grad:
            try:
                log_message(logger, f"""
                {n}: {p.grad.data.abs().mean().item():.4f}+-({p.grad.data.abs().std().item():.4f}
                """, args)
            except:
                log_message(logger, f"""
                {n}: failed - no grad
                """, args)