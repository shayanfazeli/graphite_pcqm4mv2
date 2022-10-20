import copy
import os
import os.path as osp
import rdkit
import shutil
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from graphite.utilities.logging import get_logger


logger = get_logger(__name__)
class PCQM4Mv2DatasetFull(InMemoryDataset):
    def __init__(
        self,
        root="dataset",
        smiles2graph=smiles2graph,
        transform=None,
        pre_transform=None,
        split_dict_filepath=None,
        descriptor=False,
        fingerprint=False,
        conformers_memmap: str = None,
        conformer_pool_size: int = 0,
        fingerprint_memmap: str = None,
        descriptor_memmap: str = None,
    ):
        """
        Pytorch Geometric PCQM4Mv2 dataset object
            - root (str): the dataset folder will be located at `root/pcqm4m_kddcup2021`
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed
        """

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, "pcqm4m-v2")
        self.version = 1
        self.split_dict_filepath = split_dict_filepath
        self.descriptor = descriptor
        self.fingerprint = fingerprint

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = (
            "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"
        )

        # check version and update if necessary
        if osp.isdir(self.folder) and (
            not osp.exists(osp.join(self.folder, f"RELEASE_v{self.version}.txt"))
        ):
            print("PCQM4Mv2 dataset has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

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

        super(PCQM4Mv2DatasetFull, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

        # - trying a sample get as an additional sanity check
        _ = self.get(0)


    def delete_data(self):
        self.data = None
        self.slices = None

    def load_data(self):
        self.data, self.slices = torch.load(self.processed_paths[0])

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
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "data.csv.gz"))
        smiles_list = data_df["smiles"]
        homolumogap_list = data_df["homolumogap"]

        print("there is pre-transform!")
        print("Converting SMILES strings into graphs...")
        from graphite.contrib.kpgt.data.descriptors.rdDescriptors import RDKit2D
        from graphite.contrib.kpgt.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

        data_list = []
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

            if self.include_positions:
                data.positions_3d = torch.from_numpy(np.array(
                    self.conformers_memmap[i, np.random.choice(self.conformer_pool_size), :data.num_nodes, :]))

            if self.descriptor:
                mol = rdkit.Chem.MolFromSmiles(smiles)
                data['fingerprint'] = torch.tensor(rdkit.Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)).long()
            if self.fingerprint:
                data['molecule_descriptor'] = torch.tensor(RDKit2DNormalized().process(smiles)).float()

            if self.pre_transform is not None and i == 0:
                # - testing pre-transform
                _ = self.pre_transform(copy.deepcopy(data))

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


if __name__ == "__main__":
    dataset = PCQM4Mv2DatasetFull()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())