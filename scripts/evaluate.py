"""Compute validation loss and perplexity for a checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/step_005000.pt
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.checkpointing import load_checkpoint   # noqa: E402
from training.dataset import TokenDataset            # noqa: E402
from utils.device_utils import get_device            # noqa: E402


def evaluate(checkpoint_path: str, val_path: str, batch_size: int = 32):
    device = get_device()
    cfg, model, _, _ = load_checkpoint(checkpoint_path, device)
    model.eval()

    ds = TokenDataset(val_path, cfg.block_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=True)

    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            total += F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1)).item()
            n += 1

    avg = total / max(n, 1)
    print(f"Val Loss:       {avg:.4f}")
    print(f"Val Perplexity: {math.exp(avg):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_path", default="data/processed/val.bin")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.val_path, args.batch_size)
