import numpy as np
import os
import pickle
from pathlib import Path
import argparse
import pandas as pd
import time
from tqdm import tqdm
from featurizer.rdkit_featurizer import Featurizer
import torch
import concurrent
from concurrent.futures import ProcessPoolExecutor

from multiprocessing import freeze_support, Pool, Lock, current_process, Manager
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

class CustomTimer:
    def __init__(self, path=None):
        self.path = path
        self.timers = {}
        self.f = None
        if self.path:
            self.f = open(self.path, 'w')

    def start_counter(self, key: str):
        self.timers[key] = time.perf_counter()

    def end_counter(self, key: str, msg: str):
        end = time.perf_counter()
        start = self.timers.get(key)
        if self.f and start:
            self.f.write(f'{msg}: {end - start:.2f}\n')

    def maybe_close(self):
        if self.f:
            self.f.close()
parser = argparse.ArgumentParser("Cache Features 2D/3D")

parser.add_argument("--df-path", type=Path, default=".", help="Path to PCQM4Mv2 Dataset")

parser.add_argument("--split-path", type=Path, default=".", help="Path to `split_dict.pt`")

parser.add_argument("--output-dir",
                    type=Path,
                    help="Path to store the features.",
                    default=Path("./"))

parser.add_argument("--descriptor2d",
                    type=Path,
                    help="path to descriptor 2d memap.")

parser.add_argument("--descriptor3d",
                    type=Path,
                    help="path to descriptor 2d memap.")

parser.add_argument("--conformers",
                    type=Path,
                    help="path to conformers memap.")

parser.add_argument("--sdf-path",
                    action="store_true",
                    help="Path to store the features.",
                    default=Path("./"))

parser.add_argument("--batch-size",
                    type=int,
                    help="batch size")

parser.add_argument("--num-threads",
                    type=int,
                    help="num workers to use for decriptor extraction",
                    default=1)

parser.add_argument("--num-workers",
                    type=int,
                    help="num workers to use for decriptor extraction",
                    default=40)

parser.add_argument("--run-idx",
                    type=int,
                    help="num workers to use for decriptor extraction",
                    default=0)

parser.add_argument("--total-run",
                    type=int,
                    help="num workers to use for decriptor extraction",
                    default=1)
parser.add_argument("--split",
                    type=str,
                    help="split to run on must be one of [all, train, valid, test-dev, test-challenge]",
                    default="all")

SPLITS = {"all", "train", "valid", "test-dev", "test-challenge"}

if __name__ == "__main__":
    args = parser.parse_args()
    df_path = args.df_path
    batch_size = getattr(args, 'batch_size', None)
    featurizer = Featurizer(num_threads=args.num_threads)

    splits = torch.load(args.split_path)
    df = pd.read_csv(df_path)
    humolumo_gaps = df['homolumogap']
    smiles = df['smiles']

    if args.split == "all":
        pass
    else:
        assert args.split in SPLITS, f"split must be one of {SPLITS}"
        smiles = smiles[splits[args.split]]

    conformer_path = getattr(args, "conformers", None)
    conformer_memmap, descriptor2d_memmap, descriptor3d_memmap = None, None, None
    if conformer_path is not None:
        conformer_memmap = np.memmap(
            conformer_path,
            mode='r',
            dtype='float32',
            shape=(len(smiles), 5, 51, 3)
        )
    descriptor2d_memmap = getattr(args, "descriptor2d", None)
    if descriptor2d_memmap is not None:
        descriptor2d_memmap = np.memmap(
            descriptor2d_memmap,
            mode='r',
            dtype='float32',
            shape=(len(smiles), 409)
        )

    descriptor3d_memmap = getattr(args, "descriptor3d", None)
    if descriptor3d_memmap is not None:
        descriptor3d_memmap = np.memmap(
            descriptor3d_memmap,
            # mode='r',
            dtype='float32',
            shape=(len(smiles), 5, 301)
        )

    featurizer = Featurizer(num_threads=args.num_threads,
                            # conformer_memmap=conformer_memmap,
                            # descriptor2d_memmap=descriptor2d_memmap,
                            # descriptor3d_memmap=descriptor3d_memmap,
                            generate_descriptor2d=True,
                            generate_descriptor3d=True,
                            generate_conformer=True,
                            num_conformers=5)

    smiles = smiles.to_list()
    num_smiles = len(smiles)
    start_idx = int((num_smiles // args.total_run) * args.run_idx)
    end_idx = int(min(num_smiles / args.total_run * (args.run_idx + 1), num_smiles))
    if batch_size is None:
        batch_size = int(num_smiles // args.num_workers)
    pool_split = np.arange(int(start_idx), int(end_idx), batch_size)
    total = end_idx - start_idx
    print(f"RUNNING SPLIT RANGE: {start_idx}, {end_idx} total: {total}")
    def batched_smile():
        for idx in pool_split:
            yield smiles[idx: min(end_idx, idx + batch_size)], None, idx

    timer = CustomTimer("generation_time.txt")
    timer.start_counter(f"TESTING GENERATION TIME {args.split}")
    all_data = []
    with Pool(args.num_workers, initializer=None, initargs=None) as pool:
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
                data.y = torch.Tensor([humolumo_gaps[i]])
                for d in data_dict:
                    data[d] = torch.from_numpy(data_dict[d])
                i += 1
                data_list.append(data)
                pbar.update(1)
    timer.end_counter(f"TESTING GENERATION TIME {args.split}", "TEST_GEN_TIME")
    class DummyDataset(InMemoryDataset):
        def process2(self, data_list):
            data, slices = self.collate(data_list)
            return data, slices

        def raw_file_names(self):
            return 'data.csv.gz'

        @property
        def processed_file_names(self):
            return 'place_holder.pt'
    pcqm = DummyDataset()
    data, slices = pcqm.process2(data_list)
    data_path = Path(args.df_path).parent
    dump_path = data_path /  f'feat_{args.run_idx}.pt'
    print('Saving...')
    torch.save((data, slices), str(dump_path))
