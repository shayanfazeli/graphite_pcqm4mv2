import torch
import numpy
from graphite.data.pcqm4mv2.pyg import PCQM4Mv2DatasetFull, PCQM4Mv23DDataset
from graphite.utilities.logging import get_logger

logger = get_logger(__name__)


class MultiviewPCQM4Mv2Dataset(PCQM4Mv2DatasetFull):
    """
    Parameters
    ----------

    """
    def __init__(
        self,
        num_views: int,
        transform,
        **kwargs
    ):
        """constructor"""
        super(MultiviewPCQM4Mv2Dataset, self).__init__(transform=transform, **kwargs)
        self.num_views = num_views
        assert transform is not None, "Since you have requested a multiview dataset, the provided `transform` must " \
                                      "not be None and must be capable of generating non-semantic-altering views."

    def __getitem__(self, idx):
        if (isinstance(idx, (int, numpy.integer))
                or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
                or (isinstance(idx, numpy.ndarray) and numpy.isscalar(idx))):
            return [super(MultiviewPCQM4Mv2Dataset, self).__getitem__(idx) for _ in range(self.num_views)]
        else:
            return self.index_select(idx)



class MultiviewPCQM4Mv23DDataset(PCQM4Mv23DDataset):
    """
    Parameters
    ----------

    """
    def __init__(
        self,
        num_views: int,
        transform,
        **kwargs
    ):
        """constructor"""
        super(MultiviewPCQM4Mv23DDataset, self).__init__(transform=transform, **kwargs)
        self.num_views = num_views
        assert transform is not None, "Since you have requested a multiview dataset, the provided `transform` must " \
                                      "not be None and must be capable of generating non-semantic-altering views."

    def __getitem__(self, idx):
        if (isinstance(idx, (int, numpy.integer))
                or (isinstance(idx, torch.Tensor) and idx.dim() == 0)
                or (isinstance(idx, numpy.ndarray) and numpy.isscalar(idx))):
            return [super(MultiviewPCQM4Mv23DDataset, self).__getitem__(idx) for _ in range(self.num_views)]
        else:
            return self.index_select(idx)

