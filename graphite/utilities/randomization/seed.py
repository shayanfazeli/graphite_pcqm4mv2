import random
import os
import torch
import torch.backends.cudnn
import numpy
import torch.backends.cudnn as cudnn


def fix_random_seeds(seed: int) -> None:
    """
    Parameters
    ----------
    seed: `int`, required
        The integer seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    # cudnn.benchmark = True
    # cudnn.deterministic = False
    # torch.use_deterministic_algorithms(True)
