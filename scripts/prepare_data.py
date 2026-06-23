"""Tokenize a raw text file into train/val token streams.

Builds a character-level vocabulary, encodes the whole corpus, and writes the
token IDs as flat uint16 .bin files (memmap-friendly) plus the tokenizer vocab.

Usage:
    python scripts/prepare_data.py --input data/raw/shakespeare.txt \
        --output data/processed/
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer.char_tokenizer import CharTokenizer  # noqa: E402


def prepare(input_path: str, output_dir: str, val_fraction: float = 0.1):
    text = Path(input_path).read_text(encoding="utf-8")
    print(f"Dataset size: {len(text):,} characters")

    tok = CharTokenizer(text=text)
    print(f"Vocabulary size: {tok.vocab_size}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tok.save(str(output / "tokenizer.json"))

    ids = np.array(tok.encode(text), dtype=np.uint16)

    n = len(ids)
    split = int(n * (1 - val_fraction))
    train_ids = ids[:split]
    val_ids = ids[split:]

    train_ids.tofile(output / "train.bin")
    val_ids.tofile(output / "val.bin")

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens:   {len(val_ids):,}")
    print(f"Saved to:     {output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()
    prepare(args.input, args.output, args.val_fraction)
