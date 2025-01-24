import os
import torch
from torch.utils.data import Dataset

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, block_size):
        self.block_size = block_size
        self.sequences = []

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:
                tokens = line.split()
                self.sequences.append(tokens)

        self.vocab = sorted(list({token for seq in self.sequences for token in seq}))
        self.vocab_size = len(self.vocab)
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for i, token in enumerate(self.vocab)}

        self.windows = []
        for seq in self.sequences:
            seq_len = len(seq)
            if seq_len >= block_size:
                for i in range(seq_len - block_size + 1):
                    self.windows.append((seq, i))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq, start_idx = self.windows[idx]
        chunk = seq[start_idx : start_idx + self.block_size]
        x = torch.tensor([self.stoi[token] for token in chunk[:-1]], dtype=torch.long)
        y = torch.tensor([self.stoi[token] for token in chunk[1:]], dtype=torch.long)
        return x, y