from typing import Optional
import numpy as np
from functools import lru_cache
import pyximport

pyximport.install(setup_args={'include_dirs': np.get_include()})

import os
import shutil
import os.path as osp
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.utils import smiles2graph
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from multiprocessing import Pool

from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)
from rdkit import Chem
import tarfile

class PygPCQM4Mv2FeaturizedDataset(InMemoryDataset):
    def __init__(self, 
                 featurizer,
                 root: str = 'dataset',  
                 transform = None, 
                 pre_transform = None, 
                 include_sdf: bool = False,
                 num_workers: int = 1,
                 batch_size: Optional[int] = None,
                 **kwargs):
        
        '''
            Pytorch Geometric Featurized PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - featurizer (object): configured `Featurizer` object
                - num_workers (int): number of workers to use for multiprocessing pool
        '''

        self.original_root = root
        self.featurizer = featurizer
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1
        self.batch_size = batch_size

        self.num_workers = num_workers
        self.include_sdf = include_sdf
        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'
        self.pos_url = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2FeaturizedDataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'featurized_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

        if decide_download(self.pos_url):
            path = download_url(self.pos_url, self.original_root)
            tar = tarfile.open(path, 'r:gz')
            filenames = tar.getnames()
            for file in filenames:
                tar.extract(file, self.original_root)
            tar.close()
            os.unlink(path)
        else:
            print('Stop download')
            exit(-1)


    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles'].to_list()
        homolumogap_list = data_df['homolumogap'].to_list()
        featurizer = self.featurizer
        total = len(data_df)

        if self.batch_size is None:
            batch_size = max(1, total // self.num_workers)
        else:
            batch_size = self.batch_size

        def batched_smile():
            for start_idx in range(0, total, batch_size):
                yield smiles_list[start_idx: min(total, start_idx + batch_size)], None, start_idx

        all_data = []
        with Pool(self.num_workers, initializer=None, initargs=None) as pool:
                result = list(pool.starmap(featurizer.smiles_to_graph, batched_smile()))
                pool.close()
                pool.join()
        # - flatten
        data_list = []
        i = 0
        with tqdm(total=total) as pbar:
            for batch_data_dict in result:
                for data_dict in batch_data_dict:
                    data = Data()
                    num_nodes = data_dict.pop('num_atoms')
                    data.__num_nodes__ = int(num_nodes)
                    data.y = torch.Tensor([homolumogap_list[i]])
                    for d in data_dict:
                        data[d] = torch.from_numpy(data_dict[d])
                    i += 1
                    data_list.append(data)
                    pbar.update(1)
        if self.include_sdf:
            sdf_mols = Chem.SDMolSupplier(osp.join(self.original_root, 'pcqm4m-v2-train.sdf'))
            sdf_data_list = []
            total = len(sdf_mols)

            if self.batch_size is None:
                batch_size = max(1, total // self.num_workers)
            else:
                batch_size = self.batch_size
            def batched_mol():
                for start_idx in range(0, total, batch_size):
                    mols = []
                    for i in range(min(start_idx + batch_size, total)):
                        mols.append(sdf_mols[start_idx + i])
                    yield None, mols,  start_idx
            with Pool(self.num_workers, initializer=None, initargs=None) as pool:
                result = list(pool.starmap(featurizer.smiles_to_graph, batched_mol()))
                pool.close()
                pool.join()

            i = 0
            with tqdm(total=total) as pbar:
                for batch_data_dict in result:
                    for data_dict in batch_data_dict:
                        data = Data()
                        num_nodes = data_dict.pop('num_atoms')
                        data.__num_nodes__ = int(num_nodes)
                        data.y = torch.Tensor([homolumogap_list[i]])
                        for d in data_dict:
                            data[d] = torch.from_numpy(data_dict[d])
                        i += 1
                        sdf_data_list.append(data)
                        pbar.update(1)
    
    
            data_list = sdf_data_list + data_list#[len(sdf_data_list):]
            
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        if self.include_sdf:
            split_dict['train'] = torch.arange(0, 2 * len(split_dict['train']))
            for k in split_dict:
                if k != 'train':
                    split_dict[k] = split_dict[k] + len(split_dict['train']) // 2

        return split_dict
