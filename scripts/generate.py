"""Generate text from a trained checkpoint.

Two decode paths:
  * default      : simple sliding-window forward pass (can exceed block_size by
                   truncating context); always available.
  * --use_cache  : KV-cached decode (faster, bounded by block_size).

Usage:
    python scripts/generate.py --checkpoint checkpoints/step_005000.pt \
        --prompt "ROMEO:" --tokens 300 --temperature 0.8 --top_k 40
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.kv_cache import generate_cached       # noqa: E402
from inference.sampler import sample_token           # noqa: E402
from tokenizer.char_tokenizer import CharTokenizer   # noqa: E402
from training.checkpointing import load_checkpoint   # noqa: E402
from utils.device_utils import get_device            # noqa: E402


@torch.no_grad()
def generate_simple(model, ids, max_tokens, temperature, top_k, top_p):
    """Sliding-window decode: re-runs a full forward each step, no cache."""
    model.eval()
    block_size = model.cfg.block_size
    for _ in range(max_tokens):
        context = ids[:, -block_size:]
        logits, _ = model(context)
        next_token = sample_token(logits[0, -1, :], temperature, top_k, top_p)
        ids = torch.cat([ids, next_token.view(1, 1)], dim=1)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--use_cache", action="store_true",
                        help="use KV-cached decode (bounded by block_size)")
    args = parser.parse_args()

    device = get_device()
    cfg, model, _, _ = load_checkpoint(args.checkpoint, device)
    tok = CharTokenizer(vocab_path=args.tokenizer)

    ids = torch.tensor(tok.encode(args.prompt), dtype=torch.long, device=device).unsqueeze(0)

    if args.use_cache:
        out = generate_cached(model, ids, args.tokens,
                              args.temperature, args.top_k, args.top_p)
    else:
        out = generate_simple(model, ids, args.tokens,
                              args.temperature, args.top_k, args.top_p)

    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
