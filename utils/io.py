import json
import numpy as np

def load_summary(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_embeddings(path):
    data = np.load(path)

    # Case: npz with multiple embeddings
    if isinstance(data, np.lib.npyio.NpzFile):

        # Sort keys numerically: ['1','2',...]
        keys = sorted(data.files, key=lambda x: int(x))

        # Stack embeddings → (num_shots, dim)
        embeddings = np.array([data[k] for k in keys])

        return embeddings

    return data