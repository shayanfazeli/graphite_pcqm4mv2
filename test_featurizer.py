import pandas as pd
import time
from tqdm import tqdm
from featurizer.rdkit_featurizer import Featurizer
import torch
import concurrent
from concurrent.futures import ProcessPoolExecutor

from multiprocessing import freeze_support, Pool, Lock, current_process, Manager

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

if __name__ == "__main__":
    df_path = '/workspace/data/data.csv.gz'
    featurizer = Featurizer(num_threads=32)
    timer = CustomTimer("generation_time.txt")
    timer.start_counter("TESTING GENERATION TIME")
    splits = torch.load('./split_dict.pt')
    test_split = splits['test-challenge']
    df = pd.read_csv(df_path)
    smiles = df['smiles'].values
    test_smiles = smiles[test_split]
    batch_size = 100
    size = len(test_split)
    featurizer = Featurizer(num_threads=1)
    all_data = []

    num_procs = 40
    batch_size = 10
    def batched_smile():
        for i in range(0, len(test_split), batch_size):
            yield test_smiles[i:i+batch_size]
    with Pool(num_procs, initializer=None, initargs=None) as pool:
            pool = list(tqdm(pool.imap(featurizer.smiles_to_graph, batched_smile()), total=len(test_split)//batch_size)) # initial global dictionary
            pool.close()
            pool.join()
    #with concurrent.futures.ProcessPoolExecutor(10) as executor:
    #    futures = [executor.submit(featurizer.smiles_to_graph, s) for s in test_smiles]
    #    for future in concurrent.futures.as_completed(futures):
    #        _ = future.result()



    import pdb; pdb.set_trace()
    timer.end_counter("TESTING GENERATION TIME", "TEST_GEN_TIME")
    print("don't for get to save .. ")
    import pdb; pdb.set_trace()
