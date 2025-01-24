import os
import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_path, block_size):
        self.block_size = block_size

        # Load data as tokens (words like "n60", "d4")
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Split each line into tokens
        self.tokens = []
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                self.tokens.extend(line.split())

        # Build vocabulary
        self.vocab = sorted(list(set(self.tokens)))
        self.vocab_size = len(self.vocab)
        self.stoi = {token: i for i, token in enumerate(self.vocab)}
        self.itos = {i: token for i, token in enumerate(self.vocab)}
        
        # Encode tokens to integers
        self.data = [self.stoi[token] for token in self.tokens]

    def __len__(self):
        # Number of possible sequences of length block_size
        return max(0, len(self.data) - self.block_size + 1)

    def __getitem__(self, idx):
        # Return a block of block_size tokens (x=input, y=target)
        chunk = self.data[idx : idx + self.block_size]
        x = torch.tensor(chunk[:-1], dtype=torch.long)  # Input (up to last token)
        y = torch.tensor(chunk[1:], dtype=torch.long)   # Target (shifted by 1)
        return x, y