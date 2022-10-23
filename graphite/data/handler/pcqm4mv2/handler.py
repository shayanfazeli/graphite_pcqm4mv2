from typing import Dict, List, Any
import numpy
import torch.utils.data.dataloader
import torchvision.transforms
import graphite.data.pcqm4mv2.pyg.collator as collate_fn_lib
from graphite.data.handler.base import DataHandlerBase
from graphite.data.pcqm4mv2.pyg import PCQM4Mv2Dataset, PCQM4Mv23DDataset, \
    PCQM4Mv2DatasetFull, MultiviewPCQM4Mv2Dataset, MultiviewPCQM4Mv23DDataset
import graphite.data.pcqm4mv2.pyg.transforms as transforms_lib

dataset_classes = {
    'PCQM4Mv2Dataset': PCQM4Mv2Dataset,
    'PCQM4Mv23DDataset': PCQM4Mv23DDataset,
    'PCQM4Mv2DatasetFull': PCQM4Mv2DatasetFull,
    'MultiviewPCQM4Mv2Dataset': MultiviewPCQM4Mv2Dataset,
    'MultiviewPCQM4Mv23DDataset': MultiviewPCQM4Mv23DDataset
}


class Pyg2DPCQM4Mv2(DataHandlerBase):
    """

    The idea behind data handler is that you would need only a single main configuration
    to provide the script with everything it needs to build the corresponding set of data streams,
    including instantiating the datasets and also providing the samplers.

    distributed_mode: `str`, optional (default='all')
        options are `all` and `train_only`.
    """
    def __init__(
            self,
            batch_size: int,
            root_dir: str,
            dataset: str,
            transform_configs: List[Dict[str, Any]],
            distributed: bool,
            distributed_sampling: str,
            dataloader_base_args: Dict[str, Any],
            split_dict_filepath: str = None,
            dataset_args=dict(),
            collate_fn='collate_fn',
            overfit_on_train_subset: int = None
    ):
        super(Pyg2DPCQM4Mv2, self).__init__()
        self.root_dir = root_dir  # '/home/shayan/from_source/GRPE/data'
        self.batch_size = batch_size
        self.dataset_class = dataset
        self.transform_configs = transform_configs
        self.split_dict_filepath = split_dict_filepath
        self.distributed = distributed
        self.distributed_sampling = distributed_sampling
        self.dataloader_base_args = dataloader_base_args
        self.dataset_args = dataset_args
        self.collater_choice = collate_fn
        self.overfit_on_train_subset = overfit_on_train_subset

    def get_dataloaders(self,):
        # torch.multiprocessing.freeze_support()
        dataset = dataset_classes[self.dataset_class](
            root=self.root_dir,
            split_dict_filepath=self.split_dict_filepath,
            transform=torchvision.transforms.Compose([
                getattr(transforms_lib, e['type'])(**e.get('args', dict())) for e in self.transform_configs
            ]) if len(self.transform_configs) > 0 else None,
            **self.dataset_args
        )

        split_idx = dataset.get_idx_split()
        if self.overfit_on_train_subset is not None:
            split_idx['train'] = numpy.random.choice(split_idx['train'], self.overfit_on_train_subset)

        datasets = {k: dataset[v] for k, v in split_idx.items()}

        samplers = dict()
        for mode in datasets:
            if self.distributed:
                if 'train' in mode:
                    samplers[mode] = torch.utils.data.DistributedSampler(datasets[mode])
                else:
                    if self.distributed_sampling == 'train_only':
                        samplers[mode] = torch.utils.data.SequentialSampler(datasets[mode])
                    elif self.distributed_sampling == 'all':
                        samplers[mode] = torch.utils.data.DistributedSampler(datasets[mode])
                    else:
                        raise ValueError(f'unknown value for distributed_sampling arg: {self.distributed_sampling}')
            else:
                if 'train' in mode:
                    samplers[mode] = torch.utils.data.RandomSampler(datasets[mode])
                else:
                    samplers[mode] = torch.utils.data.SequentialSampler(datasets[mode])

        dataloaders = {
            mode: torch.utils.data.DataLoader(
                dataset,
                # shuffle='train' in mode,
                batch_size=self.batch_size,
                sampler=samplers[mode],
                collate_fn=getattr(collate_fn_lib, self.collater_choice),
                **self.dataloader_base_args
            ) for mode, dataset in datasets.items()
        }
        return dataloaders
