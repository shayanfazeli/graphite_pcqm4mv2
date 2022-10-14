import sys

from .featurizer.rdkit_featurizer import Featurizer
from .featurizer.pyg_pcqm4mv2_dataset import PygPCQM4Mv2Dataset

if __name__ == "__main__":
    root = '/workspace/data/dataset'
    if len(sys.argv) >= 2:
        root_path = sys.argv[1]
    featurizer_obj = Featurizer()
    dataset = PygPCQM4Mv2Dataset(featurizer_obj.smiles_to_graph, root=root)
