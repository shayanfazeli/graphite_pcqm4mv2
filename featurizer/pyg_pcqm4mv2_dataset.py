import os
import os.path as osp
import shutil
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import tarfile

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

class PygPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, 
                 smiles2graph, 
                 root = 'dataset', 
                 transform=None, 
                 pre_transform = None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1
        
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'
        self.sdv_file = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz'
        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)
        
        if decide_download(self.sdv_file):
            path = download_url(self.sdv_file, self.original_root)
            f = tarfile.open(path)
            f.extractall(self.original_root)
            f.close()
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def process(self, batch_size=1):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles'].tolist()
        homolumogap_list = data_df['homolumogap'].tolist()

        print('Converting SMILES strings into graphs...')
        data_list = []
        for i in tqdm(range(len(smiles_list) // batch_size)):
            smiles = smiles_list[i*batch_size: min((i+1)*batch_size, len(smiles_list))]
            homolumogaps = homolumogap_list[i*batch_size:min((i+1)*batch_size, len(smiles_list))]
            graph = self.smiles2graph(smiles)
            if not isinstance(graph, list):
                graph = [graph]

            for j in range(len(graph)):
                g = graph[j]
                humolumogap = homolumogaps[j]
                if g is None:
                    print(f"could not process batch idx {i} item {j}")
                    continue
                data = Data()
                data.__num_nodes__ = int(g['num_atoms'])

                for k,v in g.items():
                    if isinstance(v, list):
                        v = np.asarray(v)
                        data[k] = torch.from_numpy(v)

                    elif isinstance(v, float):
                        data[k] = v
                    elif isinstance(v, int):
                        data[k] = v
                    elif isinstance(v, np.ndarray):
                        data[k] = torch.from_numpy(v)
                    else:
                        import pdb; pdb.set_trace()
                data.y = torch.Tensor([humolumogap])
                data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split()
        # assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        # assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        # assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        # assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        return split_dict