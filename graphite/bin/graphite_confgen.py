import pandas
import argparse
import os
import sys
import numpy
import numpy as np
import datamol as dm
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from tqdm import tqdm
import warnings
from functools import partial
from multiprocessing import freeze_support, Pool, Lock, current_process
from rdkit import RDLogger
import concurrent.futures

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')


NUM_CONFORMERS=10
ITEMCOUNT = 3746620


def write_smiles_list(start_index):
    dl = range(start_index, min(ITEMCOUNT, start_index + window_length))
    if current_process()._identity[0] == 1:
        dl = tqdm(dl)

    for i in dl:
        # - rough check for whether or not confs are generated already for this index
        if dataset[i, :, :, :].any():
            continue

        if i % 100 == 0:
            lock.acquire()
            dataset.flush()
            lock.release()
        mol = Chem.MolFromSmiles(smiles_list[i])
        try:
            new_mol = dm.conformers.generate(
                mol,
                align_conformers=True,
                n_confs=NUM_CONFORMERS,
                num_threads=16,
                minimize_energy=True,
                ignore_failure=True,
                energy_iterations=100,
            )
        except Exception as e:
            print(f"ERROR IN COMPUTING RESULTS FOR INDEX: {i}\n\terror: [{str(e)}]\n")
            continue

        # - checking datamol output
        mol = new_mol if new_mol is not None else mol

        # - reading the conformers
        conformer_positions = [e.GetPositions() for e in mol.GetConformers()][:NUM_CONFORMERS]

        if len(conformer_positions) == 0:
            continue
        else:
            conformer_positions = np.stack(conformer_positions)
            dataset[i, :conformer_positions.shape[0], :conformer_positions.shape[1], :] = conformer_positions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--df', type=str, required=True, help="the path to the `data.csv.gz` file in the PCQM4Mv2 dataset which contains smiles strings.")
    parser.add_argument('--output_filepath', type=str, required=True, help="""
    The output would be a memmap
    """)
    parser.add_argument('--num_workers', type=int, default=32, help="number_of_workers")
    args = parser.parse_args()

    lock = Lock()

    data_df = pandas.read_csv(args.df)
    smiles_list = data_df["smiles"]
    homolumogap_list = data_df["homolumogap"]

    mode = 'r+' if os.path.exists(args.output_filepath) else 'w+'

    dataset = np.memmap(
        args.output_filepath,
        dtype='float32',
        mode=mode,
        shape=(3746620, 10, 60, 3)
    )

    print("starting...")
    num_threads = args.num_workers

    window_length = ITEMCOUNT // num_threads
    start_indices = numpy.arange(0, ITEMCOUNT, window_length).tolist()

    with concurrent.futures.ProcessPoolExecutor(args.num_workers) as executor:
        # executor.map(write_smiles_list, start_indices)
        futures = [executor.submit(write_smiles_list, i) for i in start_indices]
        for future in concurrent.futures.as_completed(futures):
            _ = future.result()
