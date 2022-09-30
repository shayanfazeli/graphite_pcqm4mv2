from typing import Any
import gzip, pickle


def write_pkl_gz(data, filepath) -> None:
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(data, f)


def read_pkl_gz(filepath) -> Any:
    with gzip.open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data
