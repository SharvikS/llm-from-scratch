import json
from pathlib import Path


class CharTokenizer:
    """Character-level tokenizer.

    Build a vocabulary from raw text, then encode/decode losslessly. The vocab
    is simply the sorted set of unique characters in the corpus.
    """

    def __init__(self, text: str | None = None, vocab_path: str | None = None):
        if vocab_path and Path(vocab_path).exists():
            self._load(vocab_path)
        elif text is not None:
            self._build(text)
        else:
            raise ValueError("Provide either `text` or an existing `vocab_path`")

    def _build(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def _load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.stoi = data["stoi"]
        # JSON keys are strings; itos keys must be ints
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.vocab_size = len(self.stoi)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
