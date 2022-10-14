import pandas
import argparse
import sys
import numpy
import numpy as np
import datamol as dm
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from tqdm import tqdm
import tensorstore as ts
import warnings
from functools import partial
from multiprocessing import freeze_support, Pool, Lock, current_process
from rdkit import RDLogger 
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from .datamol_conformer import generate
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings('ignore')

ITEMCOUNT = 3746620

def write_smiles_list(start_index):
    dl = range(start_index, min(ITEMCOUNT, start_index+window_length))
        
    if current_process()._identity[0] == 1:
        dl = tqdm(dl)
    
    for i in dl:
        if dataset[i, :, :, :].any():
            continue
        
        
        if i % 100 == 0:
            lock.acquire()
            dataset.flush()
            lock.release()
        mol = Chem.MolFromSmiles(smiles_list[i])
        try:
            new_mol = generate(
                mol, 
                align_conformers=True,
                n_confs=10,
                num_threads=16,
                minimize_energy=False,
                # ignore_failure=True,
                add_hs=True,
                energy_iterations=10,
            )
        except:
            print(f"ERROR IN COMPUTING RESULTS FOR INDEX: {i}\n")
            continue
            
        mol = new_mol if new_mol is not None else mol
        
        conformer_positions = [e.GetPositions() for e in mol.GetConformers()][:50]
        
        if len(conformer_positions) == 0:
            continue
        else:
            conformer_positions = np.stack(conformer_positions)
            dataset[i,:conformer_positions.shape[0], :conformer_positions.shape[1], :] = conformer_positions
            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df', type=str, default='./data/pcqm4m-v2/raw/data.csv.gz')
    parser.add_argument('--out', type=str, default='./conformers2.npy')
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()
    
    lock = Lock()
    
    data_df = pandas.read_csv(args.df)
    smiles_list = data_df["smiles"]
    homolumogap_list = data_df["homolumogap"]
                
    dataset = np.memmap(
        args.out, 
        dtype='float32', 
        mode='w+', 
        shape=(3746620, 10, 60, 3)
    )

    print("starting...")
    num_threads=args.num_workers
    
    window_length = ITEMCOUNT//num_threads
    start_indices = numpy.arange(0,ITEMCOUNT, window_length).tolist()

    with concurrent.futures.ProcessPoolExecutor(args.num_workers) as executor:
        # executor.map(write_smiles_list, start_indices)
        futures = [executor.submit(write_smiles_list, i) for i in start_indices]
        for future in concurrent.futures.as_completed(futures):
            _ = future.result()