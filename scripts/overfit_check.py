"""Sanity check: a correct model must overfit a single fixed batch to ~0 loss.

If loss does not collapse, there is a bug in the model (residuals, masking,
gradient flow) — fix it before attempting a real training run.

Usage:
    python scripts/overfit_check.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig          # noqa: E402
from model.transformer import Transformer     # noqa: E402
from utils.device_utils import get_device     # noqa: E402


def main():
    device = get_device()
    print(f"Device: {device}")

    cfg = ModelConfig(
        vocab_size=65, d_model=64, n_heads=4, n_layers=2,
        d_ff=256, block_size=32, dropout=0.0,
    )
    model = Transformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    torch.manual_seed(0)
    x = torch.randint(0, cfg.vocab_size, (4, 32), device=device)
    y = torch.randint(0, cfg.vocab_size, (4, 32), device=device)

    initial_loss = None
    loss = None
    for step in range(200):
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            print(f"Step {step:3d} | loss {loss.item():.4f}")

    print(f"\nInitial loss: {initial_loss:.4f} | Final loss: {loss.item():.4f}")
    assert loss.item() < 0.1, "Model failed to overfit one batch — check gradient flow"
    print("Overfit check PASSED")


if __name__ == "__main__":
    main()
