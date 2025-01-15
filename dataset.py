import os

class TextDataset:
    def __init__(self, data_path, block_size):
        self.block_size = block_size

        # Load the data from the file
        with open(data_path, 'r', encoding='utf-8') as f:
            data = f.read()

        # Create vocabulary and encode data
        self.chars = sorted(list(set(data)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.data = [self.stoi[ch] for ch in data]

    def get_split(self, split_type, split_ratio):
        """
        Splits the dataset into train/val.
        :param split_type: 'train' or 'val'
        :param split_ratio: Ratio for splitting the dataset.
        :return: Data chunk for the split type.
        """
        split_idx = int(len(self.data) * split_ratio)
        if split_type == 'train':
            return self.data[:split_idx]
        elif split_type == 'val':
            return self.data[split_idx:]
        else:
            raise ValueError("split_type must be 'train' or 'val'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx:idx + self.block_size]
