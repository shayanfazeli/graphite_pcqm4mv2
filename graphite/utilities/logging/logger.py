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
