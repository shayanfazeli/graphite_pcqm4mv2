import os
import os.path as osp
import rdkit
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import extract_zip
import pandas as pd
import numpy as np
import datamol as dm
from md5checker import make_hash
from tqdm import tqdm
import torch
from dgl.data.utils import download
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from graphite.utilities.logging import get_logger
from graphite.utilities.io.extract_tar import extract_tar_gz_file
from graphite.utilities.rdkit import add_conformers
from graphite.utilities.ogb import mol2graph

logger = get_logger(__name__)


class PCQM4Mv23DDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            url_2d: str = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip',
            url_3d: str = "http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz",
            smiles2graph=smiles2graph,
            verbose=True,
            transform=None,
            pre_transform=None,
            split_dict_filepath=None,
            descriptor=False,
            fingerprint=False,
            conformers_memmap: str = None,
            conformer_pool_size: int = 1,
            fingerprint_memmap: str = None,
            descriptor_memmap: str = None,
    ):
        """
        Pytorch Geometric PCQM4Mv2 dataset object
            - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed

            # todo: add support for radius_graph from `torch_cluster`
        """
        self.verbose = verbose
        self.original_root_dir = root
        root_dir = osp.join(root, 'pcqm4m-v2')
        self.root_dir = root_dir
        self.smiles2graph = smiles2graph
        self.folder = root_dir
        self.version = 1
        self.split_dict_filepath = split_dict_filepath
        self.descriptor = descriptor
        self.fingerprint = fingerprint
        self._url_2d = url_2d
        self._url_3d = url_3d

        self.include_positions = conformer_pool_size > 0 and conformers_memmap is not None
        self.conformers_memmap = np.memmap(
            conformers_memmap,
            dtype='float32',
            mode='r',
            shape=(3746620, 10, 60, 3)
        ) if conformers_memmap is not None else None
        self.conformer_pool_size = conformer_pool_size
        assert self.conformer_pool_size <= 10, "up to 10 conformers are supported at the moment"

        self.fingerprint_memmap = np.memmap(
            fingerprint_memmap,
            dtype='float32',
            mode='r',
            shape=(3746620, 512)
        ) if fingerprint_memmap is not None else None

        self.descriptor_memmap = np.memmap(
            descriptor_memmap,
            dtype='float32',
            mode='r',
            shape=(3746620, 201)
        ) if descriptor_memmap is not None else None

        super(PCQM4Mv23DDataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def url_2d(self):
        return self._url_2d

    @property
    def url_3d(self):
        return self._url_3d

    def maybe_log(self, msg: str, level="info") -> None:
        if self.verbose:
            getattr(logger, level)(msg)

    def delete_data(self):
        self.data = None
        self.slices = None

    def load_data(self):
        self.data, self.slices = torch.load(self.processed_paths[0])

    def __getitem__(self, idx):
        if self.data is None:
            self.load_data()

        return super().__getitem__(idx)

    def get(self, idx):
        g = super().get(idx)

        if self.fingerprint:
            if self.fingerprint_memmap is None and 'fingerprint' not in g:
                raise Exception("~> the cached dataset, as is, does not contain fingerprint information."
                                "please consider recreating the dataset or to provide fingerprint memmap.")

            if 'fingerprint' not in g:
                g['fingerprint'] = torch.from_numpy(np.array(self.fingerprint_memmap[idx, :])).float()

        if self.descriptor:
            if self.descriptor_memmap is None and 'molecule_descriptor' not in g:
                raise Exception("~> the cached dataset, as is, does not contain molecule descriptor information."
                                "please consider recreating the dataset or to provide molecule descriptor memmap.")

            if 'molecule_descriptor' not in g:
                g['molecule_descriptor'] = torch.from_numpy(np.array(self.descriptor_memmap[idx, :])).float()

        if self.include_positions and 'positions_3d' not in g:
            g.positions_3d = torch.from_numpy(np.array(self.conformers_memmap[idx, np.random.choice(self.conformer_pool_size), :g.num_nodes, :]))
        return g

    @property
    def raw_file_names(self):
        return "data.csv.gz"

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

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
        data_df = pd.read_csv(osp.join(self.raw_dir, "data.csv.gz"))
        # - in the 3d dataset, we wont use the smiles (for train) and use the
        # sdf instead.
        smiles_list = data_df["smiles"]
        homolumogap_list = data_df["homolumogap"]

        # - reading the 3d file
        print("reading the SDF file...")
        suppl = rdkit.Chem.SDMolSupplier(osp.join(self.raw_dir, 'pcqm4m-v2-train.sdf'))

        print("Converting SMILES strings into graphs...")
        from graphite.contrib.kpgt.data.descriptors.rdDescriptors import RDKit2D
        from graphite.contrib.kpgt.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
        data_list = []
        train_indices = set(self.get_idx_split()['train'].tolist())
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]

            # - even for 3d data, we would still
            # use the bonds instead of `radius_graph`
            if i in train_indices:
                mol = suppl[i]
            else:
                mol = rdkit.Chem.MolFromSmiles(smiles)
            graph = mol2graph(mol)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([homolumogap])

            # 3d graph for train indices
            if i in train_indices:
                # - storing the positions_3d
                positions = torch.from_numpy(mol.GetConformer(0).GetPositions()).float()
                assert data.num_nodes == positions.shape[0]
                data['positions_3d'] = positions
            else:
                data['positions_3d'] = torch.from_numpy(np.array(
                    self.conformers_memmap[i, np.random.choice(self.conformer_pool_size), :data.num_nodes, :]))

            if self.descriptor:
                mol = rdkit.Chem.MolFromSmiles(smiles)
                data['fingerprint'] = torch.tensor(rdkit.Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)).long()
            if self.fingerprint:
                data['molecule_descriptor'] = torch.tensor(RDKit2DNormalized().process(smiles)).float()

            if self.pre_transform is not None and i == 0:
                # - testing pre-transform
                _ = self.pre_transform(data)

            data_list.append(data)

        # double-check prediction target
        split_dict = self.get_idx_split()
        assert all([not torch.isnan(data_list[i].y)[0] for i in split_dict["train"]])
        assert all([not torch.isnan(data_list[i].y)[0] for i in split_dict["valid"]])
        assert all([torch.isnan(data_list[i].y)[0] for i in split_dict["test-dev"]])
        assert all(
            [torch.isnan(data_list[i].y)[0] for i in split_dict["test-challenge"]]
        )

        if self.pre_transform is not None:
            print("applying pre-processing transform...")
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        data, slices = self.collate(data_list)

        print("Saving...")
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        if self.split_dict_filepath is None:
            split_dict = replace_numpy_with_torchtensor(
                torch.load(osp.join(self.root, "split_dict.pt"))
            )
        else:
            split_dict = replace_numpy_with_torchtensor(
                torch.load(self.split_dict_filepath)
            )
        return split_dict
