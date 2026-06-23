"""Train the Transformer from a YAML config.

Usage:
    python scripts/train.py --config configs/tiny.yaml
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.transformer import Transformer          # noqa: E402
from training.dataset import build_dataloaders      # noqa: E402
from training.trainer import Trainer                # noqa: E402
from utils.config_loader import load_config         # noqa: E402
from utils.count_params import count_params         # noqa: E402
from utils.device_utils import get_device           # noqa: E402


def plot_loss_curves(train_losses, val_losses, out_path="notebooks/loss_curves.png"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(train_losses, alpha=0.7, linewidth=0.8)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-Entropy Loss")

    if val_losses:
        steps, losses = zip(*val_losses)
        ppl = [math.exp(l) for l in losses]
        axes[1].plot(steps, ppl, marker="o", linewidth=1.5)
        axes[1].set_title("Validation Perplexity")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Perplexity")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Loss curves saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--checkpoint_dir", default="checkpoints/")
    parser.add_argument("--train_bin", default="data/processed/train.bin")
    parser.add_argument("--val_bin", default="data/processed/val.bin")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Config: {args.config}")
    print(f"Device: {get_device()}")

    model = Transformer(cfg)
    count_params(model)

    train_loader, val_loader = build_dataloaders(
        train_path=args.train_bin,
        val_path=args.val_bin,
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
    )

    trainer = Trainer(model, cfg, train_loader, val_loader, args.checkpoint_dir)
    trainer.train()
    plot_loss_curves(trainer.train_losses, trainer.val_losses)


if __name__ == "__main__":
    main()
