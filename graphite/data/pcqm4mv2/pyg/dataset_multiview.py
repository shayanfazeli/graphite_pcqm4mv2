import torch
from graphite.data.pcqm4mv2.pyg import PCQM4Mv2DatasetFull
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
        return [super().__getitem__(idx) for _ in range(self.num_views)]
