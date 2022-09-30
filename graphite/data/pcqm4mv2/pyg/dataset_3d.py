import os
import os.path as osp
import rdkit
import shutil
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
import datamol as dm
from md5checker import make_hash
from tqdm import tqdm
import torch
from dgl.data.utils import download
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from graphite.contrib.kpgt.data.descriptors.rdDescriptors import RDKit2D
from graphite.contrib.kpgt.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from graphite.utilities.logging import get_logger
from graphite.utilities.io.extract_tar import extract_tar_gz_file
from graphite.utilities.rdkit import add_conformers

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
            additional_conformer_count: int = 0
    ):
        """
        Pytorch Geometric PCQM4Mv2 dataset object
            - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed
        """
        self.verbose = verbose
        self.original_root_dir = root
        root_dir = osp.join(root, 'pcqm4m-v2')
        self.root_dir = root_dir
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "pcqm4m-v2")
        self.version = 1
        self.split_dict_filepath = split_dict_filepath
        self.descriptor = descriptor
        self.fingerprint = fingerprint
        self._url_2d = url_2d
        self._url_3d = url_3d
        self.additional_conformer_count = additional_conformer_count

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
        smiles_list = data_df["smiles"]
        homolumogap_list = data_df["homolumogap"]

        # - reading the 3d file
        print("reading the SDF file...")
        suppl = rdkit.Chem.SDMolSupplier(osp.join(self.raw_dir, 'pcqm4m-v2-train.sdf'))

        print("Converting SMILES strings into graphs...")
        data_list = []
        train_indices = set(self.get_idx_split()['train'].tolist())
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            graph = self.smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            data.__num_nodes__ = int(graph["num_nodes"])
            data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
            data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
            data.y = torch.Tensor([homolumogap])

            # 3d graph for train indices
            if i in train_indices:
                mol = suppl[i]
                positions = torch.from_numpy(mol.GetConformer(0).GetPositions()).float()
                assert data.num_nodes == positions.shape[0]
                data['positions_3d'] = positions
                if self.additional_conformer_count > 0:
                    mol = dm.conformers.generate(mol, align_conformers=True)
                    conformers = mol.GetConformers()
                    for conf_index, conformer in enumerate(conformers):
                        positions = torch.from_numpy(conformer.GetPositions()).float()
                        assert data.num_nodes == positions.shape[0]
                        data[f'positions_3d_{conf_index}'] = positions

            if self.descriptor:
                mol = rdkit.Chem.MolFromSmiles(smiles)
                data['fingerprint'] = torch.tensor(rdkit.Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)).long()
            if self.fingerprint:
                data['molecule_descriptor'] = torch.tensor(RDKit2DNormalized().process(smiles)).float()
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
            data_list = [self.pre_transform(data) for data in data_list]

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


