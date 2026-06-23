"""Render attention heatmaps for a prompt through a trained checkpoint.

Usage:
    python scripts/visualize.py --checkpoint checkpoints/step_005000.pt \
        --prompt "To be or not"
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer.char_tokenizer import CharTokenizer    # noqa: E402
from training.checkpointing import load_checkpoint    # noqa: E402
from utils.device_utils import get_device             # noqa: E402
from utils.visualize_attention import plot_attention_heads  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    parser.add_argument("--prompt", default="To be or not")
    parser.add_argument("--out", default="notebooks/attention_map.png")
    args = parser.parse_args()

    device = get_device()
    cfg, model, _, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    tok = CharTokenizer(vocab_path=args.tokenizer)

    ids = torch.tensor(tok.encode(args.prompt), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        _, attn_weights = model(ids)

    tokens = list(args.prompt)  # char-level: one token per character
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plot_attention_heads(attn_weights, tokens, save_path=args.out)


if __name__ == "__main__":
    main()
