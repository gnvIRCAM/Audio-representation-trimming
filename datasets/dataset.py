from random import random

import lmdb
import numpy as np
import torch
from tqdm import tqdm

from .audio_example import AudioExample


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path,
        keys=['waveform', 'metadata'],
        readonly=True,
    ) -> None:
        super().__init__()

        self.env = lmdb.open(
            path,
            lock=False,
            readonly=readonly,
            readahead=False,
            map_async=False,
        )
        with self.env.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
        self.buffer_keys = keys
        

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index=None):

        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[index]))
        out = {}
        
        for key in self.buffer_keys:
            if key == "metadata":
                out[key] = ae.get_metadata()
            else:
                try:
                    out[key] = ae.get(key)
                except:
                    print("key: ", key, " not found")
        return out