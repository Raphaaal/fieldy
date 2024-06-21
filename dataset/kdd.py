from torch.utils.data import Dataset
import torch
import pandas as pd

class KDDDataset(Dataset):
    def __init__(
            self,
            vocab_file="../data/kdd/vocab.nb",
            samples=None,
            targets=None,
            vocab=None,
            ncols=None, 
            seq_len=None, 
            data=None,
            data_root=None,
            dry_run=0,
        ):
        self.samples = samples
        self.targets = targets
        self.vocab = vocab
        self.ncols = ncols
        self.seq_len = seq_len
        self.data = data
        self.data_root = data_root
        self.dry_run = dry_run

    def __getitem__(self, index):
        input_seq = torch.tensor(
            self.samples[index], 
            dtype=torch.long
        )
        label = torch.tensor([self.targets[index]])

        return (input_seq, label)

    def __len__(self):
        if self.dry_run:
            self.samples = self.samples[:self.dry_run]
        return len(self.samples)