"""Train a BPE tokenizer from a raw text file.

Usage:
    python tokenizer/train_tokenizer.py --input data/raw/shakespeare.txt \
        --vocab_size 1000 --output data/processed/bpe_tokenizer.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer.bpe_tokenizer import BPETokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--output", default="data/processed/bpe_tokenizer.json")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    tok = BPETokenizer()
    tok.train(text, vocab_size=args.vocab_size)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    tok.save(args.output)
    print(f"BPE vocab size: {tok.vocab_size}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
