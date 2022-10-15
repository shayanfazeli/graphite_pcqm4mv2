import pandas as pd
import time
from tqdm import tqdm
from featurizer.rdkit_featurizer import Featurizer
import torch
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
    batch_size = 100
    size = len(test_split)
    featurizer = Featurizer(num_threads=32)
    tq = tqdm(range(size // batch_size + 1))
    all_data = []
    for idx in tq:
        idxs = test_split[idx * batch_size:(idx + 1)*batch_size]
        batch_smiles = smiles[idxs]
        data = featurizer.smiles_to_graph(batch_smiles)
        all_data.append(data)
    timer.end_counter("TESTING GENERATION TIME", "TEST_GEN_TIME")
    print("don't for get to save .. ")
    import pdb; pdb.set_trace()
