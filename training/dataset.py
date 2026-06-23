import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TokenDataset(Dataset):
    """Samples fixed-length context windows from a flat token .bin file.

    Each item is a (x, y) pair where y is x shifted by one position — the
    standard next-token-prediction target. Uses memmap so the file is never
    loaded fully into RAM.
    """

    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        # number of valid starting positions for a (block_size + 1) window
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def build_dataloaders(
    train_path: str,
    val_path: str,
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
):
    train_ds = TokenDataset(train_path, block_size)
    val_ds = TokenDataset(val_path, block_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader, val_loader
