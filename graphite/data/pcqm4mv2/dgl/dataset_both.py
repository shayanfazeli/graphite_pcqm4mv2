from typing import List, Dict, Callable
import os
import os.path as osp
from tqdm import tqdm

from md5checker import make_hash

import pandas
import torch
import torch.nn

import dgl
from dgl.data import DGLDataset
from dgl.data.utils import download, get_download_dir, load_graphs, save_graphs, Subset
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import extract_zip, download_url, decide_download
import rdkit
import rdkit.Chem

from graphite.data.pcqm4mv2.dgl.transforms.base import BaseGraphitePCQM4MTransform
from graphite.data.pcqm4mv2.dgl.transforms.edge.edge_encoding import EncodeEdgeType
from graphite.data.pcqm4mv2.dgl.transforms.node.add_task_node import AddTaskNode
from graphite.data.pcqm4mv2.dgl.transforms.node.node2node_connection_type_encoding import EncodeNode2NodeConnectionType
from graphite.data.pcqm4mv2.dgl.transforms.node.node2node_shortest_path_encoding import \
    EncodeNode2NodeShortestPathFeatureTrajectory
from graphite.data.pcqm4mv2.dgl.transforms.node.node2node_shortest_path_length_type_encoding import \
    EncodeNode2NodeShortestPathLengthType
from graphite.data.pcqm4mv2.dgl.transforms.node.node_type_encoding import EncodeNodeType
from graphite.utilities.io.extract_tar import extract_tar_gz_file
from graphite.utilities.logging import get_logger

logger = get_logger(__name__)


class PCQM4Mv2Dataset(DGLDataset):
    def __init__(
            self,
            root_dir: str,
            cache_name: str = 'dataset',
            url_2d: str = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip',
            url_3d: str = "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz",
            smiles2graph: Callable = smiles2graph,
            verbose: bool = False,
            force_reload: bool = False,
            preprocessing_transforms: List[BaseGraphitePCQM4MTransform] = [
                EncodeNode2NodeShortestPathFeatureTrajectory(),
                EncodeNodeType(),
                AddTaskNode(),
                EncodeEdgeType(),
                EncodeNode2NodeConnectionType(),
                EncodeNode2NodeShortestPathLengthType()
            ],
            transforms: List[BaseGraphitePCQM4MTransform] = [
                EncodeNode2NodeShortestPathFeatureTrajectory(),
                EncodeNodeType(),
                AddTaskNode(),
                EncodeEdgeType(),
                EncodeNode2NodeConnectionType(),
                EncodeNode2NodeShortestPathLengthType()
            ],
            caching_mode: str = 'all_in_one'
    ):
        assert caching_mode in ['all_in_one', 'one_in_one'], f"unknown caching mode: {caching_mode}"
        self.caching_mode = caching_mode
        self.cache_name = cache_name
        self.original_root_dir = root_dir
        root_dir = osp.join(root_dir, 'pcqm4m-v2')
        self.root_dir = root_dir
        self._url_2d = url_2d
        self._url_3d = url_3d
        self.smiles2graph = smiles2graph
        self.preprocessing_transforms = preprocessing_transforms
        self.transforms = transforms
        super(PCQM4Mv2Dataset, self).__init__(
            name='pcqm4mv2',
            url=None,  # overriding it
            raw_dir=osp.join(root_dir, 'raw'),
            save_dir=osp.join(root_dir, 'processed'),
            hash_key=(),
            force_reload=force_reload,
            verbose=verbose,
            transform=None
        )

    @property
    def url_2d(self):
        return self._url_2d

    @property
    def url_3d(self):
        return self._url_3d

    def maybe_log(self, msg: str, level="info") -> None:
        if self.verbose:
            getattr(logger, level)(msg)

    def download(self):
        self.maybe_log(msg="downloading raw material...")

        # - the molecules + labels (2d info from smiles)
        self.maybe_log(msg="downloading the 2d information on the molecules...")

        download(self.url_2d, self.original_root_dir)
        # assert make_hash(osp.join(self.raw_dir, 'pcqm4m-v2-train.sdf.tar.gz')) == ''

        self.maybe_log(msg="extracting the 2d information...")
        extract_zip(osp.join(self.original_root_dir, self.url_2d.rpartition('/')[2]), self.original_root_dir)
        os.unlink(osp.join(self.original_root_dir, self.url_2d.rpartition('/')[2]))

        # - the 3d information
        self.maybe_log(msg="downloading the 3d information on the atom positionings...")
        download(self.url_3d, self.raw_dir)
        assert make_hash(osp.join(self.raw_dir, 'pcqm4m-v2-train.sdf.tar.gz')) == 'fd72bce606e7ddf36c2a832badeec6ab'

        self.maybe_log(msg="extracting 3d info [sdf file]...")
        extract_tar_gz_file(f"{osp.join(self.raw_dir, 'pcqm4m-v2-train.sdf.tar.gz')}")

    def process(self):
        raw_dir = self.raw_dir
        processed_dir = self.save_dir
        preprocessed_filepath = osp.join(self.save_dir, 'dgl_data_processed.bin')

        # - if cached
        if osp.exists(preprocessed_filepath):
            self.maybe_log(msg="loading preprocessed graphs")
            self.graphs, label_dict = load_graphs(preprocessed_filepath)
            self.labels = label_dict['labels']  # homo-lumo gaps (ev)
        # - if not cached
        else:
            # - reading the 2d file
            data_df = pandas.read_csv(osp.join(raw_dir, 'data.csv.gz'))

            # - reading the 3d file
            suppl = rdkit.Chem.SDMolSupplier(osp.join(self.raw_dir, 'pcqm4m-v2-train.sdf'))

            smiles_list = data_df['smiles']
            homolumogap_list = data_df['homolumogap']

            self.maybe_log(msg='Converting SMILES strings into graphs...')
            self.graphs = []
            self.graphs_3d = []
            self.labels = []

            train_indices = set(self.get_idx_split()['train'].tolist())
            for i in tqdm(range(len(smiles_list))):
                smiles = smiles_list[i]
                homolumogap = homolumogap_list[i]
                graph = self.smiles2graph(smiles)

                # - 3d info

                assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])

                # assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
                assert (len(graph['node_feat']) == graph['num_nodes'])

                dgl_graph = dgl.graph((graph['edge_index'][0], graph['edge_index'][1]), num_nodes=graph['num_nodes'])
                dgl_graph.edata['feat'] = torch.from_numpy(graph['edge_feat']).to(torch.int64)
                dgl_graph.ndata['feat'] = torch.from_numpy(graph['node_feat']).to(torch.int64)

                for transform in self.preprocessing_transforms:
                    dgl_graph = transform(dgl_graph)

                # 3d graph for train indices
                # if i in train_indices:
                #     mol = suppl[i]
                #     positions = torch.from_numpy(mol.GetConformer(0).GetPositions()).to(
                #         dgl_graph.ndata['feat'].device).float()
                #     assert dgl_graph.num_nodes() == positions.shape[0]
                #     dgl_graph.ndata['position'] = positions

                if self.caching_mode == 'all_in_one':
                    self.graphs.append(dgl_graph)
                    self.graphs_3d.append(dgl.DGLGraph())
                    self.labels.append(homolumogap)
                elif self.caching_mode == 'one_in_one':
                    filepath = osp.join(self.save_dir, f'item_{i}_dgl_dataset_{self.cache_name}_processed.bin')
                    save_graphs(filepath, [dgl_graph, dgl.DGLGraph()], labels={'labels': torch.tensor([homolumogap, homolumogap]).long()})

            if self.caching_mode == 'all_in_one':
                self.labels = torch.tensor(self.labels, dtype=torch.float32)

            # double-check prediction target
            split_dict = self.get_idx_split()

            if self.caching_mode == 'all_in_one':
                assert (all([not torch.isnan(self.labels[i]) for i in split_dict['train']]))
                assert (all([not torch.isnan(self.labels[i]) for i in split_dict['valid']]))
                assert (all([torch.isnan(self.labels[i]) for i in split_dict['test-dev']]))
                assert (all([torch.isnan(self.labels[i]) for i in split_dict['test-challenge']]))

    def save(self):
        if self.caching_mode == 'all_in_one':
            self.maybe_log('Saving all in one...')
            preprocessed_filepath = osp.join(self.save_dir, f'dgl_dataset_{self.cache_name}_processed.bin')
            preprocessed_filepath_3d = osp.join(self.save_dir, f'dgl_dataset_{self.cache_name}_processed_3d.bin')
            save_graphs(preprocessed_filepath, self.graphs, labels={'labels': self.labels})
            save_graphs(preprocessed_filepath_3d, self.graphs_3d, labels={'labels': self.labels})

    def load(self):
        if self.caching_mode == 'all_in_one':
            preprocessed_filepath = osp.join(self.save_dir, f'dgl_dataset_{self.cache_name}_processed.bin')

            # - if cached
            if osp.exists(preprocessed_filepath):
                self.maybe_log(msg="loading preprocessed 2d graphs")
                self.graphs, label_dict = load_graphs(preprocessed_filepath)
                self.labels = label_dict['labels']  # homo-lumo gaps (ev)

            preprocessed_filepath_3d = osp.join(self.save_dir, f'dgl_dataset_{self.cache_name}_processed_3d.bin')

            # - if cached
            if osp.exists(preprocessed_filepath):
                self.maybe_log(msg="loading preprocessed 3d graphs")
                self.graphs_3d, _ = load_graphs(preprocessed_filepath_3d)
        else:
            pass


    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root_dir, 'split_dict.pt')))
        return split_dict

    def has_cache(self):
        if self.caching_mode == 'all_in_one':
            paths = [
                osp.join(self.save_dir, f'dgl_dataset_{self.cache_name}_processed.bin'),
                osp.join(self.root_dir, 'split_dict.pt')
            ]
        else:
            paths = [osp.join(self.root_dir, 'split_dict.pt')]
            paths += [osp.join(self.save_dir, f'item_{i}_dgl_dataset_{self.cache_name}_processed.bin') for i in range(1000)]

        return all([osp.exists(e) for e in paths])

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, int):
            if self.caching_mode == 'all_in_one':
                g, y = self.graphs[idx], self.labels[idx]
                for t in self.transforms:
                    g = t(g)
                return g, y
            elif self.caching_mode == 'one_in_one':
                graph, label = load_graphs(osp.join(self.save_dir, f'item_{idx}_dgl_dataset_{self.cache_name}_processed.bin'))
                graph = graph[0]
                label = label['labels'][0]
                return graph, label
            else:
                raise NotImplementedError()
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            if idx.dim() == 0:
                return self.graphs[idx], self.labels[idx]
            elif idx.dim() == 1:
                return Subset(self, idx.cpu())

        raise IndexError(
            'Only integers and long are valid '
            'indices (got {}).'.format(type(idx).__name__))

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.graphs)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))

# # Collate function for ordinary graph classification
# def collate_dgl(samples):
#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)
#
#     if isinstance(labels[0], torch.Tensor):
#         return batched_graph, torch.stack(labels)
#     else:
#         return batched_graph, labels
#
#
# if __name__ == '__main__':
#     dataset = PCQM4Mv2Dataset(root_dir='~/gt')
#     print(dataset)
#     print(dataset[100])
#     split_dict = dataset.get_idx_split()
#     print(split_dict)
#     print(dataset[split_dict['train']])
#     print(collate_dgl([dataset[0], dataset[1], dataset[2]]))
