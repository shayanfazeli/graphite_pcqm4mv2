import sys
import json
import argparse
import warnings
from functools import partial
from pathlib import Path
from collections import defaultdict
import pandas as pd
import rdkit
import numpy as np
import datamol as dm
from tqdm import tqdm
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
import datamol
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support, Pool, Lock, current_process, Manager

from .descriptors import descriptor_2d_fn_map, descriptor_3d_fn_map

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')


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

def get_MMFF_atom_poses(mol, num_confs=None):
    """the atoms of mol will be changed in some cases."""
    try:
        new_mol = Chem.AddHs(mol)
        res = AllChem.EmbedMultipleConfs(new_mol, numConfs=num_confs)
        ### MMFF generates multiple conformations
        res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
        new_mol = Chem.RemoveHs(new_mol)
        energy = np.asarray([x[1] for x in res])
    except Exception as e:
        print(f"Exception occured: {e}, falling back to 2D coordinates")
        new_mol = mol
        AllChem.Compute2DCoords(new_mol)
        energy = 0
    return new_mol, energy


def get_atom_poses(mol, conf):
    atom_poses = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
        pos = conf.GetAtomPosition(i)
        atom_poses.append([pos.x, pos.y, pos.z])
    return np.asarray(atom_poses)

def datamol_conf_gen(mol):
    new_mol = generate(
        mol, 
        align_conformers=True,
        n_confs=GLOBAL_MAX_NUM_CONFS,
        num_threads=16,
        minimize_energy=True,
        add_hs=True,
        energy_iterations=10,
    )
    return new_mol

def conf_to_mol(mol, conf_id):
    conf = mol.GetConformer(conf_id)
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(Chem.Conformer(conf))
    return new_mol

def write_smiles_list(start_index):
    dl = range(start_index, min(GLOBAL_ITEMCOUNT, start_index+GLOBAL_WINDOW_LEN))
    global GLOBAL_METADATA_2D, GLOBAL_METADATA_3D
    if current_process()._identity[0] == 1 and not GLOBAL_ARGS.compute_descriptors:
        dl = tqdm(dl)

    for i in dl:

        if GLOBAL_CONFORMER_DS[i, :, :, :].any():
            continue

        if i % 100 == 0:
            GLOBAL_LOCK.acquire()
            GLOBAL_CONFORMER_DS.flush()
            GLOBAL_ENERGY_DS.flush()
            GLOBAL_DESCRIPTORS_DS_3D.flush()
            GLOBAL_DESCRIPTORS_DS_2D.flush()
            GLOBAL_LOCK.release()

        mol = Chem.MolFromSmiles(GLOBAL_SMILES_LIST[i])
        if GLOBAL_ARGS.compute_descriptors:
            descriptors_2d = datamol.descriptors.batch_compute_many_descriptors([mol],
                            properties_fn=descriptor_2d_fn_map,
                            add_properties=False,
                            batch_size=1,
                            n_jobs=GLOBAL_ARGS.num_workers_descriptors)
            col_num = 0
            for k in GLOBAL_METADATA_2D:
                dtype = descriptors_2d[k].dtype
                v = descriptors_2d[k]
                GLOBAL_METADATA_2D, val = update_metadata_val(GLOBAL_METADATA_2D, k, v, dtype)
                l = GLOBAL_METADATA_2D[k]["len"]
                GLOBAL_DESCRIPTORS_DS_2D[i, col_num:col_num + l] = val.reshape(-1, l)
                col_num += l
        energy = None
        try:
            # if GLOBAL_ARGS.conformer_generation == "rdkit":
            new_mol, energy = get_MMFF_atom_poses(mol, num_confs=int(GLOBAL_MAX_NUM_CONFS))
            # elif GLOBAL_ARGS.conformer_generation == "datamol":
            #     new_mol = datamol_conf_gen(mol)
        except:
            print(f"ERROR IN COMPUTING RESULTS FOR INDEX: {i}\n")
            continue
     
        mol = new_mol if new_mol is not None else mol
        
        if GLOBAL_ARGS.compute_descriptors: 
            batch_mol = [conf_to_mol(mol, conf.GetId()) for conf in mol.GetConformers()]
            num_samples = len(batch_mol)
            descriptors_3d = datamol.descriptors.batch_compute_many_descriptors(batch_mol,
                                properties_fn=descriptor_3d_fn_map,
                                add_properties=False,
                                batch_size=num_samples,
                                n_jobs=GLOBAL_ARGS.num_workers_descriptors)
            col_num = 0
            for k in GLOBAL_METADATA_3D:
                dtype = descriptors_3d[k].dtype
                v = descriptors_3d[k]
                GLOBAL_METADATA_3D, val = update_metadata_val(GLOBAL_METADATA_3D, k, v, dtype)
                l = GLOBAL_METADATA_3D[k]["len"]
                GLOBAL_DESCRIPTORS_DS_3D[i, :num_samples, col_num:col_num + l] = val.reshape(-1, l)
                col_num += l

        conformer_positions = [get_atom_poses(mol, conf) for conf in mol.GetConformers()]
        if len(conformer_positions) == 0:
            print("This Shouldn't Be Hapenning...! :( ")
        else:
            conformer_positions = np.stack(conformer_positions)
            GLOBAL_CONFORMER_DS[i, :conformer_positions.shape[0], :conformer_positions.shape[1], :] = np.asarray(conformer_positions)
            if GLOBAL_ARGS.conformer_generation == "rdkit":
                GLOBAL_ENERGY_DS[i,:conformer_positions.shape[0]] = np.asarray(energy)
    return (1, dl)

def init_pool(*batch_smiles):
    global GLOBAL_METADATA_2D, GLOBAL_METADATA_3D
    batch_mol = [Chem.MolFromSmiles(smiles) for smiles in batch_smiles]
    num_samples = len(batch_mol)
    df_2d = datamol.descriptors.batch_compute_many_descriptors(batch_mol,
            properties_fn=descriptor_2d_fn_map,
            add_properties=False,
            batch_size=num_samples,
            n_jobs=GLOBAL_ARGS.num_workers_descriptors)
    new_batch_mol = []
    for mol in batch_mol:
         new_mol, energy = get_MMFF_atom_poses(mol, num_confs=2)
         new_batch_mol.append(new_mol)
    num_samples = len(new_batch_mol)
    # _= [AllChem.Compute2DCoords(new_mol) for new_mol in batch_mol]
    df_3d = datamol.descriptors.batch_compute_many_descriptors(new_batch_mol,
                properties_fn=descriptor_3d_fn_map,
                add_properties=False,
                batch_size=num_samples,
                n_jobs=GLOBAL_ARGS.num_workers_descriptors)
    GLOBAL_METADATA_3D = init_metadata(df_3d)
    GLOBAL_METADATA_2D = init_metadata(df_2d)

    assert sum([GLOBAL_METADATA_2D[k]["len"] for k in GLOBAL_METADATA_2D]) == GLOBAL_NUM_DESCRIPTOR_FEATURES_2D, "NUM_DESCRIPTOR_DIM DOES NOT MATCH"
    assert sum([GLOBAL_METADATA_3D[k]["len"] for k in GLOBAL_METADATA_3D]) == GLOBAL_NUM_DESCRIPTOR_FEATURES_3D, "NUM_DESCRIPTOR_DIM DOES NOT MATCH"
    

def main(args):
    global GLOBAL_LOCK, GLOBAL_SMILES_LIST, GLOBAL_CONFORMER_DS, GLOBAL_ENERGY_DS, \
    GLOBAL_WINDOW_LEN, GLOBAL_ITEMCOUNT, GLOBAL_MAX_NUM_ATOMS, GLOBAL_MAX_NUM_CONFS, GLOBAL_ARGS, \
    GLOBAL_DESCRIPTORS_DS_3D, GLOBAL_DESCRIPTORS_DS_2D, GLOBAL_NUM_DESCRIPTOR_FEATURES_2D, GLOBAL_NUM_DESCRIPTOR_FEATURES_3D, \
    GLOBAL_METADATA_2D, GLOBAL_METADATA_3D
    GLOBAL_ARGS = args
    
    GLOBAL_LOCK = Lock()

    data_df = pd.read_csv(args.df)
    GLOBAL_SMILES_LIST = data_df["smiles"]
    GLOBAL_ITEMCOUNT = len(GLOBAL_SMILES_LIST)
    GLOBAL_MAX_NUM_ATOMS = 51
    GLOBAL_NUM_DESCRIPTOR_FEATURES_2D = 409
    GLOBAL_NUM_DESCRIPTOR_FEATURES_3D = 301
    GLOBAL_MAX_NUM_CONFS = args.num_conformers


    homolumogap_list = data_df["homolumogap"]
    out_dir = Path(args.out_dir)

    GLOBAL_DESCRIPTORS_DS_2D = np.memmap(
        out_dir / "descriptors_2d.npy",
        dtype='float32',
        mode='w+',
        shape=(GLOBAL_ITEMCOUNT, GLOBAL_NUM_DESCRIPTOR_FEATURES_2D))
    
    GLOBAL_DESCRIPTORS_DS_3D = np.memmap(
        out_dir / "descriptors_3d.npy",
        dtype='float32',
        mode='w+',
        shape=(GLOBAL_ITEMCOUNT, GLOBAL_MAX_NUM_CONFS, GLOBAL_NUM_DESCRIPTOR_FEATURES_3D))

    GLOBAL_CONFORMER_DS = np.memmap(
        out_dir / "conformers.npy",
        dtype='float32',
        mode='w+',
        shape=(GLOBAL_ITEMCOUNT, GLOBAL_MAX_NUM_CONFS, GLOBAL_MAX_NUM_ATOMS, 3)
    )

    GLOBAL_ENERGY_DS = np.memmap(
        out_dir / "energy.npy",
        dtype='float32',
        mode='w+',
        shape=(GLOBAL_ITEMCOUNT, GLOBAL_MAX_NUM_CONFS)
    )
    
    print("starting...")
    num_threads=args.num_workers

    GLOBAL_WINDOW_LEN = GLOBAL_ITEMCOUNT // num_threads
    start_indices = np.arange(0, GLOBAL_ITEMCOUNT, GLOBAL_WINDOW_LEN).tolist()

    if GLOBAL_ARGS.compute_descriptors:
        manager = Manager()
        GLOBAL_METADATA_2D = manager.dict()
        GLOBAL_METADATA_3D = manager.dict()
        smiles_batch = GLOBAL_SMILES_LIST[:100].to_list()

        with Pool(args.num_workers, initializer=init_pool, initargs=(smiles_batch)) as pool:
            GLOBAL_METADATA_2D, GLOBAL_METADATA_3D = {}, {}
            pool = list(tqdm(pool.imap(write_smiles_list, start_indices), total=len(start_indices))) # initial global dictionary
            # pool.map(write_smiles_list, start_indices) # using pool processing to call f()function
            pool.close()
            pool.join()
        meta_path = out_dir / "metadata_descriptor3d.json"
        with open(meta_path, "w") as outfile:
            json.dump(GLOBAL_METADATA_3D, outfile, cls=CustomEncoder)
        meta_path = out_dir / "metadata_descriptor2d.json"
        with open(meta_path, "w") as outfile:
            json.dump(GLOBAL_METADATA_2D, outfile, cls=CustomEncoder)
    else:
        with concurrent.futures.ProcessPoolExecutor(args.num_workers) as executor:
            # executor.map(write_smiles_list, start_indices)
            futures = [executor.submit(write_smiles_list, i) for i in start_indices]
            for future in concurrent.futures.as_completed(futures):
                _ = future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--df', type=Path, default='./data/pcqm4m-v2/raw/data.csv.gz')
    parser.add_argument('--out-dir', type=Path, default='./')
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--num-conformers', type=int, default=10)
    parser.add_argument('--num-workers-descriptors', type=int, default=16)
    parser.add_argument('--compute-descriptors', 
                        action="store_true", type=bool, default=False)
    parser.add_argument('--conformer-generation', type=str, default="rdkit",
                        help="conformer generation scheme to use must be one of [rdkit, datamol]")


    args = parser.parse_args()
    main(args)


