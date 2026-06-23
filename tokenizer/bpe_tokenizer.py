import json
from collections import Counter, defaultdict

from tqdm import tqdm


class BPETokenizer:
    """Byte-Pair Encoding tokenizer trained from scratch.

    Starts from a character vocabulary and repeatedly merges the most frequent
    adjacent pair until the target vocab size is reached. A "</w>" marker ends
    each word so decode can restore spaces.
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.merges: list[tuple[str, str]] = []
        self.itos: dict[int, str] = {}
        self.vocab_size = 0

    def train(self, text: str, vocab_size: int, verbose: bool = True):
        chars = sorted(set(text))
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.merges = []

        word_counts = Counter(text.split())
        corpus = {
            tuple(list(word) + ["</w>"]): count
            for word, count in word_counts.items()
        }

        n_merges = max(0, vocab_size - len(self.vocab))
        for _ in tqdm(range(n_merges), desc="BPE merges", disable=not verbose):
            pair_counts: dict[tuple, int] = defaultdict(int)
            for word, count in corpus.items():
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += count
            if not pair_counts:
                break

            a, b = max(pair_counts, key=pair_counts.get)
            merged = a + b
            self.vocab[merged] = len(self.vocab)
            self.itos[len(self.itos)] = merged
            self.merges.append((a, b))

            new_corpus = {}
            for word, count in corpus.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_corpus[tuple(new_word)] = count
            corpus = new_corpus

        self.vocab_size = len(self.vocab)

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        for a, b in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == a and tokens[i + 1] == b:
                    tokens = tokens[:i] + [a + b] + tokens[i + 2:]
                else:
                    i += 1
        return tokens

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            merged = self._apply_merges(list(word) + ["</w>"])
            ids.extend(self.vocab[t] for t in merged if t in self.vocab)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids).replace("</w>", " ").strip()

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {
                    "vocab": self.vocab,
                    "merges": self.merges,
                    "itos": {str(k): v for k, v in self.itos.items()},
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tok = cls()
        with open(path) as f:
            data = json.load(f)
        tok.vocab = data["vocab"]
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        tok.vocab_size = len(tok.vocab)
        return tok
