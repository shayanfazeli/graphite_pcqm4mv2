import sys
import os
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import rdkit
import datamol
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from .descriptors import descriptor_2d_fn_map, descriptor_3d_fn_map

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser("Cache Descriptors")

parser.add_argument("--data-path", type=Path, default=".", help="Path to PCQM4Mv2 Dataset")

parser.add_argument("--output-dir",
                    type=Path,
                    help="Path to store the features.",
                    default=Path("./"))

parser.add_argument("--sdf",
                    action="store_true",
                    help="Path to store the features.",
                    default=Path("./"))

parser.add_argument("--batch-size",
                    type=int,
                    help="batch size",
                    default="1000")

parser.add_argument("--num-workers",
                    type=int,
                    help="num workers to use for decriptor extraction",
                    default="32")

def init_metadata(df):
    metadata = defaultdict(dict)
    for column in df.columns:
        col = df[column]
        dtype = col.dtype
        metadata[column]["dtype"] = dtype
        if dtype == np.dtype('O'):
            l = np.asarray(col.values.tolist()).shape[-1]
            metadata[column]["len"] = l
        if dtype == np.int64:
            metadata[column]["uniques"] = set()
            metadata[column]["len"] = 1
            metadata[column]["max"] = float("inf")
            metadata[column]["min"] = float("-inf")
        elif dtype == np.float64:
            metadata[column]["max"] = float("-inf")
            metadata[column]["min"] = float("inf")
            metadata[column]["len"] = 1
    return metadata

def update_metadata_val(metadata, k, v: pd.Series, dtype):
    if dtype == np.dtype('O'):
        val = np.asarray(v.values.tolist()).astype(np.float32)
    elif dtype == np.dtype(np.float64):
        metadata[k]["max"] = max(v.max(), metadata[k]["max"])
        metadata[k]["min"] = min(v.min(), metadata[k]["min"])
        val = v.values.astype(np.float32)
    elif dtype == np.dtype(np.int64):
        metadata[k]["uniques"] = metadata[k]["uniques"].union(set(v.values))
        metadata[k]["max"] = max(v.max(), metadata[k]["max"])
        metadata[k]["min"] = min(v.min(), metadata[k]["min"])
        val = v.values.astype(np.float32)
    else:
        raise ValueError("dtype not supported")
    
    return metadata, val

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.dtype):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)



if __name__ == "__main__":
    args = parser.parse_args()
    data_path = args.data_path
    sdf_path = data_path  / "pcqm4m-v2-train.sdf"
    smiles_path = data_path / "raw" / "data.csv.gz"

    smiles_df = pd.read_csv(smiles_path)
    output_dir = args.output_dir
    npy_2d_path = output_dir / "descriptors_2d.npy"
    descriptors_2d = np.memmap(npy_2d_path, mode="w+", dtype="float32", shape=(len(smiles_df), 409))

    batch_size = args.batch_size
    num_workers = args.num_workers
#     metadata_2d = None
#     logger.info("Starting extracting 2d descriptors")
#     for i in tqdm(range(len(smiles_df) // batch_size), total=len(smiles_df)//batch_size, desc="Extracting 2d Descriptors"):
#         strings = smiles_df['smiles'].iloc[i*batch_size: min(len(smiles_df), (i+1)*batch_size)]
#         batch = []
#         for s in strings:
#             mol  = rdkit.Chem.MolFromSmiles(s)
#             batch.append(mol)
#         num_samples = len(batch)
#         df = datamol.descriptors.batch_compute_many_descriptors(batch,
#                                                                 properties_fn=descriptor_2d_fn_map,
#                                                                 add_properties=False,
#                                                                 batch_size=num_samples,
#                                                                 n_jobs=num_workers)
#         if metadata_2d is None:
#             metadata_2d = init_metadata(df)
#         col_num = 0
#         data = []
#         for k in metadata_2d:
#             dtype = df[k].dtype
#             v = df[k]
#             metadata_2d, val = update_metadata_val(metadata_2d, k, v, dtype)
#             l = metadata_2d[k]["len"]
#             descriptors_2d[i * batch_size: i * batch_size + 1 * num_samples, col_num:col_num + l] = val.reshape(-1, l)
#             col_num += l

#     meta_path = output_dir / "metadata_descriptor2.json"
#     logger.info(f"Dumping data metadata to: {meta_path}")
#     with open(meta_path, "w") as outfile:
#         json.dump(metadata_2d, outfile, cls=CustomEncoder)
#     logger.info(f"Dumping data : {npy_2d_path}")
#     descriptors_2d.flush()

    logger.info("Starting extracting 3d descriptors")
    metadata_3d = None
    suppl = rdkit.Chem.SDMolSupplier(str(sdf_path))
    descriptors_3d = np.memmap(output_dir / "descriptors_3d.npy", mode="w+", dtype="float32", shape=(len(suppl), 301))
    for idx in tqdm(range(len(suppl) // batch_size), total=len(suppl)//batch_size, desc="Extracting 3d Descriptors"):
        batch = []
        for i in range(idx * batch_size, min(len(suppl), (idx + 1) * batch_size)):
            batch.append(suppl[i])
        num_samples = len(batch)
        df = datamol.descriptors.batch_compute_many_descriptors(batch,
                                                                properties_fn=descriptor_3d_fn_map,
                                                                add_properties=False,
                                                                batch_size=num_samples,
                                                                n_jobs=num_workers)
        if metadata_3d is None:
            metadata_3d = init_metadata(df)
        col_num = 0
        data = []
        for k in metadata_3d:
            dtype = df[k].dtype
            v = df[k]
            metadata_3d, val = update_metadata_val(metadata_3d, k, v, dtype)
            l = metadata_3d[k]["len"]
            descriptors_3d[idx*batch_size: idx* batch_size + num_samples, col_num:col_num + l] = val.reshape(-1, l)
            col_num += l

    logger.info(f"Dumping data metadata to: {meta_path}")
    meta_path = output_dir / "metadata_descriptor3d.json"
    with open(meta_path, "w") as outfile:
        json.dump(metadata_3d, outfile, cls=CustomEncoder)

    logger.info(f"Dumping data : {npy_2d_path}")
    descriptors_3d.flush()
    logger.info(f"Finished caching descriptors in {output_dir}")


