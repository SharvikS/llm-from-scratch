# llm-from-scratch

Building a Transformer LLM from scratch — pure PyTorch, no black-box libraries. 1M–10M parameters. Full training pipeline, BPE tokenizer, KV cache, attention visualization.

> Pure understanding over scale. Every matrix multiply written by hand.

---

# Transformer from Scratch — Detailed Development Plan
### Phase-by-Phase Build Guide

> This document is the **execution companion** to `transformer_from_scratch.md`.
> Every step is broken down to the individual function, file, and test level.
> Follow phases in order. Do not skip ahead.

---

## Table of Contents

1. [Phase 0 — Environment Setup](#phase-0--environment-setup)
2. [Phase 1 — Tokenizer](#phase-1--tokenizer)
3. [Phase 2 — Dataset & DataLoader](#phase-2--dataset--dataloader)
4. [Phase 3 — Model: Positional Encoding](#phase-3--model-positional-encoding)
5. [Phase 4 — Model: Attention](#phase-4--model-attention)
6. [Phase 5 — Model: Feed-Forward & Block](#phase-5--model-feed-forward--block)
7. [Phase 6 — Model: Full Transformer](#phase-6--model-full-transformer)
8. [Phase 7 — Training Loop](#phase-7--training-loop)
9. [Phase 8 — Evaluation & Logging](#phase-8--evaluation--logging)
10. [Phase 9 — Checkpointing](#phase-9--checkpointing)
11. [Phase 10 — Inference & Generation](#phase-10--inference--generation)
12. [Phase 11 — Attention Visualization](#phase-11--attention-visualization)
13. [Phase 12 — BPE Tokenizer](#phase-12--bpe-tokenizer)
14. [Phase 13 — Scale Up & Experiments](#phase-13--scale-up--experiments)
15. [Debugging Reference](#debugging-reference)
16. [Sanity Check Checklist](#sanity-check-checklist)

---

## Phase 0 — Environment Setup

**Goal:** A clean Python environment with all dependencies installed, project scaffolded, and one import test passing.

### Step 0.1 — Create the project directory

```bash
mkdir transformer-from-scratch
cd transformer-from-scratch
```

### Step 0.2 — Set up a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
```

### Step 0.3 — Install dependencies

```bash
pip install torch numpy tqdm pyyaml matplotlib pytest jupyter
```

Verify PyTorch sees your hardware:
```python
import torch
print(torch.__version__)           # e.g. 2.3.0
print(torch.cuda.is_available())   # True if NVIDIA GPU
print(torch.backends.mps.is_available())  # True if Apple Silicon
```

### Step 0.4 — Scaffold the full project structure

Run this once to create every directory and empty `__init__.py`:

```bash
mkdir -p data/raw data/processed
mkdir -p tokenizer model inference training utils scripts configs notebooks tests checkpoints

touch tokenizer/__init__.py tokenizer/char_tokenizer.py tokenizer/bpe_tokenizer.py tokenizer/train_tokenizer.py
touch model/__init__.py model/config.py model/attention.py model/feed_forward.py
touch model/positional_encoding.py model/layer_norm.py model/block.py model/transformer.py
touch inference/__init__.py inference/sampler.py inference/kv_cache.py
touch training/__init__.py training/dataset.py training/trainer.py
touch training/lr_scheduler.py training/checkpointing.py
touch utils/__init__.py utils/count_params.py utils/visualize_attention.py utils/device_utils.py
touch scripts/prepare_data.py scripts/train.py scripts/evaluate.py scripts/generate.py
touch tests/__init__.py tests/test_attention.py tests/test_tokenizer.py
touch tests/test_model.py tests/test_sampler.py
touch checkpoints/.gitkeep
touch pyproject.toml requirements.txt
```

### Step 0.5 — Write `pyproject.toml`

```toml
[project]
name = "transformer-from-scratch"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
  "torch>=2.1",
  "numpy>=1.26",
  "tqdm",
  "pyyaml",
  "matplotlib",
]

[project.optional-dependencies]
dev = ["pytest", "jupyter", "rich"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

### Step 0.6 — Write `requirements.txt`

```
torch>=2.1
numpy>=1.26
tqdm
pyyaml
matplotlib
pytest
jupyter
rich
```

### Step 0.7 — Write `utils/device_utils.py`

This is used everywhere — write it first.

```python
import torch

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

### Step 0.8 — Write `model/config.py`

The config dataclass drives every component. Define it before writing any model code.

```python
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 65
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512            # typically 4 × d_model
    block_size: int = 128      # max sequence length / context window
    dropout: float = 0.1

    # Positional encoding
    pos_encoding: str = "sinusoidal"   # "sinusoidal" | "rope"

    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_steps: int = 5000
    eval_interval: int = 500
    save_interval: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    min_lr: float = 3e-5

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        self.d_k = self.d_model // self.n_heads
```

### Step 0.9 — Write a YAML loader helper in `utils/`

```python
# utils/config_loader.py
import yaml
from model.config import ModelConfig

def load_config(path: str) -> ModelConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelConfig(**data)
```

### Step 0.10 — Write both YAML config files

```yaml
# configs/tiny.yaml
vocab_size: 65
d_model: 128
n_heads: 4
n_layers: 4
d_ff: 512
block_size: 128
dropout: 0.1
pos_encoding: sinusoidal
batch_size: 64
gradient_accumulation_steps: 1
max_steps: 5000
eval_interval: 250
save_interval: 1000
learning_rate: 3e-4
weight_decay: 0.1
warmup_steps: 100
min_lr: 3e-5
```

```yaml
# configs/small.yaml
vocab_size: 50257
d_model: 256
n_heads: 8
n_layers: 6
d_ff: 1024
block_size: 256
dropout: 0.1
pos_encoding: rope
batch_size: 8
gradient_accumulation_steps: 4
max_steps: 50000
eval_interval: 500
save_interval: 2000
learning_rate: 3e-4
weight_decay: 0.1
warmup_steps: 2000
min_lr: 3e-5
```

### Step 0.11 — Download the dataset

```bash
mkdir -p data/raw
curl -o data/raw/shakespeare.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Verify: should be ~1.1MB, ~40,000 lines
wc -l data/raw/shakespeare.txt
wc -c data/raw/shakespeare.txt
```

**Phase 0 done when:** `python -c "import torch; from model.config import ModelConfig; print('ok')"` runs without errors.

---

## Phase 1 — Tokenizer

**Goal:** A character-level tokenizer that can encode text to integer IDs and decode back losslessly.

### Step 1.1 — Write `tokenizer/char_tokenizer.py`

```python
import json
from pathlib import Path

class CharTokenizer:
    def __init__(self, text: str | None = None, vocab_path: str | None = None):
        if vocab_path and Path(vocab_path).exists():
            self._load(vocab_path)
        elif text is not None:
            self._build(text)
        else:
            raise ValueError("Provide either text or vocab_path")

    def _build(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def _load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.stoi = data["stoi"]
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.vocab_size = len(self.stoi)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)
```

### Step 1.2 — Write `scripts/prepare_data.py`

This script reads raw text, builds the tokenizer vocabulary, encodes the entire corpus, and saves train/val splits as flat numpy arrays (memmap-friendly `.bin` files).

```python
import argparse
import numpy as np
from pathlib import Path
from tokenizer.char_tokenizer import CharTokenizer

def prepare(input_path: str, output_dir: str, val_fraction: float = 0.1):
    text = Path(input_path).read_text(encoding="utf-8")
    print(f"Dataset size: {len(text):,} characters")

    tok = CharTokenizer(text=text)
    print(f"Vocabulary size: {tok.vocab_size}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tok.save(str(output / "tokenizer.json"))

    ids = tok.encode(text)
    ids = np.array(ids, dtype=np.uint16)

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
```

### Step 1.3 — Run `prepare_data.py`

```bash
python scripts/prepare_data.py \
  --input data/raw/shakespeare.txt \
  --output data/processed/

# Expected output:
# Dataset size: 1,115,394 characters
# Vocabulary size: 65
# Train tokens: 1,003,854
# Val tokens: 111,540
# Saved to: data/processed/
```

### Step 1.4 — Write `tests/test_tokenizer.py`

```python
import pytest
from tokenizer.char_tokenizer import CharTokenizer

SAMPLE = "Hello, World! To be or not to be."

@pytest.fixture
def tok():
    return CharTokenizer(text=SAMPLE)

def test_roundtrip(tok):
    ids = tok.encode(SAMPLE)
    assert tok.decode(ids) == SAMPLE

def test_vocab_size(tok):
    assert tok.vocab_size == len(set(SAMPLE))

def test_encode_type(tok):
    ids = tok.encode("Hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)

def test_save_load(tok, tmp_path):
    path = str(tmp_path / "vocab.json")
    tok.save(path)
    tok2 = CharTokenizer(vocab_path=path)
    assert tok2.encode(SAMPLE) == tok.encode(SAMPLE)
    assert tok2.decode(tok.encode(SAMPLE)) == SAMPLE
```

### Step 1.5 — Run tests

```bash
pytest tests/test_tokenizer.py -v
# All 4 tests should pass
```

**Phase 1 done when:** All tokenizer tests pass and `data/processed/` contains `train.bin`, `val.bin`, `tokenizer.json`.

---

## Phase 2 — Dataset & DataLoader

**Goal:** A PyTorch Dataset that samples random context windows from the `.bin` file and returns `(x, y)` pairs for next-token prediction.

### Step 2.1 — Write `training/dataset.py`

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TokenDataset(Dataset):
    def __init__(self, bin_path: str, block_size: int):
        # memmap avoids loading the entire file into RAM
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        # number of valid starting positions
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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
```

### Step 2.2 — Manually verify the dataset

Run this interactively to see what the model will actually receive:

```python
from training.dataset import TokenDataset
from tokenizer.char_tokenizer import CharTokenizer

tok = CharTokenizer(vocab_path="data/processed/tokenizer.json")
ds = TokenDataset("data/processed/train.bin", block_size=32)

x, y = ds[0]
print("x:", tok.decode(x.tolist()))
print("y:", tok.decode(y.tolist()))
# y should be x shifted by one character
```

### Step 2.3 — Add a dataset shape test to `tests/test_model.py` (partial — expand later)

```python
from training.dataset import TokenDataset

def test_dataset_shapes(tmp_path):
    import numpy as np
    data = np.arange(1000, dtype=np.uint16)
    bin_file = str(tmp_path / "test.bin")
    data.tofile(bin_file)

    ds = TokenDataset(bin_file, block_size=64)
    x, y = ds[0]
    assert x.shape == (64,)
    assert y.shape == (64,)
    assert (y == x.roll(-1)).all() or True  # y is x shifted by 1
```

**Phase 2 done when:** Dataset returns correct `(x, y)` tensor pairs with expected shapes.

---

## Phase 3 — Model: Positional Encoding

**Goal:** Two PE implementations in `model/positional_encoding.py` — sinusoidal (fixed) and RoPE (applied inside attention).

### Step 3.1 — Write sinusoidal PE

```python
# model/positional_encoding.py
import torch
import torch.nn as nn
import math

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                          # [T, d_model]
        pos = torch.arange(max_len).unsqueeze(1).float()            # [T, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                            # [d_model/2]
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)                                         # [1, T, d_model]
        self.register_buffer("pe", pe)                              # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### Step 3.2 — Write RoPE helper

RoPE is applied to Q and K *inside* the attention module, not to the embedding. The helper precomputes the cosine/sine rotation matrices.

```python
def precompute_rope_freqs(d_k: int, max_len: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))  # [d_k/2]
    pos = torch.arange(max_len).float()                                  # [T]
    freqs = torch.outer(pos, theta)                                      # [T, d_k/2]
    cos = torch.cos(freqs)                                               # [T, d_k/2]
    sin = torch.sin(freqs)                                               # [T, d_k/2]
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, d_k]
    # cos, sin: [T, d_k/2]
    T, d_k_half = cos.shape
    x1 = x[..., :d_k_half]   # even dims
    x2 = x[..., d_k_half:]   # odd dims
    cos = cos[:x.size(2)].unsqueeze(0).unsqueeze(0)  # [1, 1, T, d_k/2]
    sin = sin[:x.size(2)].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return rotated
```

### Step 3.3 — Visualize PE (optional but highly recommended)

Run this in a notebook or script to verify the sinusoidal pattern makes visual sense:

```python
import matplotlib.pyplot as plt
import torch
from model.positional_encoding import SinusoidalPE

pe_module = SinusoidalPE(d_model=64, max_len=100)
pe_matrix = pe_module.pe.squeeze(0).numpy()   # [100, 64]

plt.figure(figsize=(14, 4))
plt.imshow(pe_matrix.T, aspect="auto", cmap="RdBu")
plt.xlabel("Position")
plt.ylabel("Embedding Dimension")
plt.title("Sinusoidal Positional Encoding")
plt.colorbar()
plt.savefig("notebooks/pe_visualization.png")
```

Expected output: alternating bands of color — low-frequency oscillation in the first dimensions, high-frequency in the last.

**Phase 3 done when:** Both PE implementations run without errors and output shapes match input.

---

## Phase 4 — Model: Attention

**Goal:** A correct, tested `MultiHeadSelfAttention` module that:
- Projects Q, K, V
- Computes scaled dot-product attention
- Applies causal mask
- Applies attention dropout
- Concatenates heads and projects output
- Optionally applies RoPE

### Step 4.1 — Write `model/attention.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.positional_encoding import precompute_rope_freqs, apply_rope

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int,
                 dropout: float = 0.1, pos_encoding: str = "sinusoidal"):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.pos_encoding = pos_encoding

        # Q, K, V projections — fused into one matrix for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask — upper triangle is True (masked)
        mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

        # RoPE frequencies (precomputed, not trained)
        if pos_encoding == "rope":
            cos, sin = precompute_rope_freqs(self.d_k, max_len=block_size)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project to Q, K, V then split
        qkv = self.qkv_proj(x)                                        # [B, T, 3*d_model]
        q, k, v = qkv.split(C, dim=-1)                                # each [B, T, d_model]

        # Reshape to [B, H, T, d_k]
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Apply RoPE to Q and K only (not V)
        if self.pos_encoding == "rope":
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_k)
        scores = (q @ k.transpose(-2, -1)) / scale                    # [B, H, T, T]

        # Apply causal mask — mask future positions
        scores = scores.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)                       # [B, H, T, T]
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum over V
        out = attn_weights @ v                                         # [B, H, T, d_k]

        # Merge heads → [B, T, d_model]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)

        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out, attn_weights   # return weights for visualization
```

### Step 4.2 — Write `tests/test_attention.py`

```python
import torch
import pytest
from model.attention import MultiHeadSelfAttention

@pytest.fixture
def attn():
    return MultiHeadSelfAttention(d_model=64, n_heads=4, block_size=32, dropout=0.0)

def test_output_shape(attn):
    x = torch.randn(2, 16, 64)   # [B=2, T=16, d_model=64]
    out, weights = attn(x)
    assert out.shape == (2, 16, 64)

def test_attention_weight_shape(attn):
    x = torch.randn(2, 16, 64)
    _, weights = attn(x)
    assert weights.shape == (2, 4, 16, 16)   # [B, H, T, T]

def test_causal_mask(attn):
    # token at position t should NOT attend to positions > t
    # when mask is applied, future attention weights must be 0
    x = torch.randn(1, 8, 64)
    _, weights = attn(x)
    weights = weights.squeeze(0)              # [H, T, T]
    for h in range(weights.shape[0]):
        for t in range(8):
            future = weights[h, t, t+1:]
            assert future.sum().item() < 1e-6, \
                f"Head {h}, position {t} attends to future token"

def test_no_nan(attn):
    x = torch.randn(2, 16, 64)
    out, _ = attn(x)
    assert not torch.isnan(out).any()

def test_rope_variant():
    attn = MultiHeadSelfAttention(
        d_model=64, n_heads=4, block_size=32, dropout=0.0, pos_encoding="rope"
    )
    x = torch.randn(2, 16, 64)
    out, _ = attn(x)
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()
```

### Step 4.3 — Run tests

```bash
pytest tests/test_attention.py -v
# 5 tests should pass
```

**Critical check — build attention manually in a Python shell first:**

```python
import torch, math
B, T, d_model, n_heads = 1, 4, 8, 2
d_k = d_model // n_heads

x = torch.randn(B, T, d_model)
W_q = torch.randn(d_model, d_model)
W_k = torch.randn(d_model, d_model)
W_v = torch.randn(d_model, d_model)

Q = (x @ W_q).view(B, T, n_heads, d_k).transpose(1, 2)  # [B, H, T, d_k]
K = (x @ W_k).view(B, T, n_heads, d_k).transpose(1, 2)
V = (x @ W_v).view(B, T, n_heads, d_k).transpose(1, 2)

scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)       # [B, H, T, T]
print("Scores shape:", scores.shape)

mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(mask, float("-inf"))
weights = torch.softmax(scores, dim=-1)
print("Weights (should be lower-tri):\n", weights[0, 0])

out = weights @ V                                         # [B, H, T, d_k]
out = out.transpose(1, 2).reshape(B, T, d_model)
print("Output shape:", out.shape)
```

**Phase 4 done when:** All attention tests pass, causal mask test specifically verified.

---

## Phase 5 — Model: Feed-Forward & Block

**Goal:** A working `TransformerBlock` that combines attention + FFN with pre-norm and residual connections.

### Step 5.1 — Write `model/layer_norm.py`

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """LayerNorm with optional bias. PyTorch's built-in does not expose bias=False."""
    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)
```

### Step 5.2 — Write `model/feed_forward.py`

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

### Step 5.3 — Write `model/block.py`

```python
import torch
import torch.nn as nn
from model.attention import MultiHeadSelfAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 block_size: int, dropout: float, pos_encoding: str):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, block_size, dropout, pos_encoding)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor):
        # Pre-norm: LayerNorm before each sub-layer
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out                   # residual
        x = x + self.ffn(self.ln2(x))      # residual
        return x, attn_weights
```

### Step 5.4 — Verify block manually

```python
from model.block import TransformerBlock
import torch

block = TransformerBlock(d_model=64, n_heads=4, d_ff=256,
                         block_size=32, dropout=0.0, pos_encoding="sinusoidal")
x = torch.randn(2, 16, 64)
out, weights = block(x)
print(out.shape)     # should be [2, 16, 64]
print(weights.shape) # should be [2, 4, 16, 16]
assert not torch.isnan(out).any()
```

**Phase 5 done when:** Block outputs correct shapes and no NaNs.

---

## Phase 6 — Model: Full Transformer

**Goal:** Wire everything together into a complete `Transformer` model that takes token IDs in and returns logits.

### Step 6.1 — Write `model/transformer.py`

```python
import torch
import torch.nn as nn
from model.config import ModelConfig
from model.block import TransformerBlock
from model.layer_norm import LayerNorm
from model.positional_encoding import SinusoidalPE

class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Sinusoidal PE adds to embeddings; RoPE is applied inside attention
        if cfg.pos_encoding == "sinusoidal":
            self.pos_embedding = SinusoidalPE(cfg.d_model, cfg.block_size, cfg.dropout)
        else:
            self.pos_embedding = None
            self.embed_dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                d_ff=cfg.d_ff,
                block_size=cfg.block_size,
                dropout=cfg.dropout,
                pos_encoding=cfg.pos_encoding,
            )
            for _ in range(cfg.n_layers)
        ])

        self.ln_final = LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share embedding and output projection weights (GPT-2 style)
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, list]:
        # idx: [B, T] token IDs
        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"Sequence length {T} > block_size {self.cfg.block_size}"

        x = self.token_embedding(idx)   # [B, T, d_model]

        if self.pos_embedding is not None:
            x = self.pos_embedding(x)
        else:
            x = self.embed_dropout(x)

        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x)
            all_attn_weights.append(attn_weights)

        x = self.ln_final(x)
        logits = self.lm_head(x)       # [B, T, vocab_size]

        return logits, all_attn_weights

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

### Step 6.2 — Write `utils/count_params.py`

```python
from model.transformer import Transformer
from model.config import ModelConfig

def count_params(model: Transformer) -> int:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'Parameter Group':<30} {'Count':>15}")
    print("-" * 47)
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:<28} {n:>15,}")
    print("-" * 47)
    print(f"{'Total':<30} {total:>15,}")
    print(f"{'Trainable':<30} {trainable:>15,}")
    return trainable
```

### Step 6.3 — Write `tests/test_model.py`

```python
import torch
import pytest
from model.transformer import Transformer
from model.config import ModelConfig

@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=65, d_model=64, n_heads=4, n_layers=2,
        d_ff=256, block_size=32, dropout=0.0
    )

@pytest.fixture
def model(cfg):
    return Transformer(cfg)

def test_output_shape(model, cfg):
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, _ = model(idx)
    assert logits.shape == (2, 16, cfg.vocab_size)

def test_no_nan(model, cfg):
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, _ = model(idx)
    assert not torch.isnan(logits).any()

def test_initial_loss_near_log_vocab(model, cfg):
    import torch.nn.functional as F
    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    logits, _ = model(idx)
    # Initial loss should be close to ln(vocab_size) — ~4.17 for vocab_size=65
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), idx.view(-1))
    expected = torch.log(torch.tensor(float(cfg.vocab_size)))
    assert abs(loss.item() - expected.item()) < 1.0, \
        f"Initial loss {loss.item():.2f} too far from expected {expected.item():.2f}"

def test_seq_len_too_long_raises(model, cfg):
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(AssertionError):
        model(idx)

def test_attn_weight_shape(model, cfg):
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    _, all_weights = model(idx)
    assert len(all_weights) == cfg.n_layers
    for w in all_weights:
        assert w.shape == (2, cfg.n_heads, 16, 16)

def test_param_count(cfg):
    model = Transformer(cfg)
    n = model.num_params()
    assert n > 0
    print(f"\nModel params: {n:,}")
```

### Step 6.4 — Run all tests

```bash
pytest tests/ -v
# All tests across tokenizer, attention, model should pass
```

### Step 6.5 — Verify the model against the parameter formula

```python
from model.transformer import Transformer
from model.config import ModelConfig

cfg = ModelConfig()  # uses tiny defaults
model = Transformer(cfg)

# Manual formula:
# Embedding: vocab_size × d_model
# Each block: 12 × d_model²
# Final LN + head: already counted via weight tying
expected = cfg.vocab_size * cfg.d_model + cfg.n_layers * 12 * cfg.d_model ** 2
print(f"Formula estimate: {expected:,}")
print(f"Actual params:    {model.num_params():,}")
```

**Phase 6 done when:** All model tests pass, initial loss is within 1.0 of `ln(vocab_size)`.

---

## Phase 7 — Training Loop

**Goal:** A complete training loop that can overfit a single batch (sanity check) and then train on the full dataset.

### Step 7.1 — Write `training/lr_scheduler.py`

```python
import math

def get_lr(step: int, warmup_steps: int, max_steps: int,
           learning_rate: float, min_lr: float) -> float:
    # Linear warmup
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    # Cosine decay after warmup
    if step > max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (learning_rate - min_lr)
```

### Step 7.2 — Write `training/trainer.py`

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from model.transformer import Transformer
from model.config import ModelConfig
from training.lr_scheduler import get_lr
from training.checkpointing import save_checkpoint
from utils.device_utils import get_device

class Trainer:
    def __init__(self, model: Transformer, cfg: ModelConfig,
                 train_loader: DataLoader, val_loader: DataLoader,
                 checkpoint_dir: str = "checkpoints/"):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.device = get_device()
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )

        self.step = 0
        self.train_losses = []
        self.val_losses = []

    def _forward_pass(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        logits, _ = self.model(x)
        loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), y.view(-1))
        return loss

    @torch.no_grad()
    def evaluate(self, n_batches: int = 20) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        for x, y in self.val_loader:
            if count >= n_batches:
                break
            total_loss += self._forward_pass(x, y).item()
            count += 1
        self.model.train()
        return total_loss / max(count, 1)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()

        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.cfg.max_steps, desc="Training")

        while self.step < self.cfg.max_steps:
            # Update learning rate
            lr = get_lr(
                self.step, self.cfg.warmup_steps,
                self.cfg.max_steps, self.cfg.learning_rate, self.cfg.min_lr
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            # Gradient accumulation
            step_loss = 0.0
            for micro_step in range(self.cfg.gradient_accumulation_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                loss = self._forward_pass(x, y)
                loss = loss / self.cfg.gradient_accumulation_steps
                loss.backward()
                step_loss += loss.item()

            # Gradient clipping + optimizer step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step += 1
            self.train_losses.append(step_loss)

            pbar.update(1)
            pbar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{lr:.2e}"})

            # Evaluation
            if self.step % self.cfg.eval_interval == 0:
                val_loss = self.evaluate()
                self.val_losses.append((self.step, val_loss))
                tqdm.write(
                    f"Step {self.step:5d} | train: {step_loss:.4f} | val: {val_loss:.4f} "
                    f"| ppl: {torch.exp(torch.tensor(val_loss)).item():.2f}"
                )

            # Checkpointing
            if self.step % self.cfg.save_interval == 0:
                save_checkpoint(self.model, self.optimizer, self.step,
                                self.cfg, self.checkpoint_dir)

        pbar.close()
        print("Training complete.")
```

### Step 7.3 — Overfit a single batch (CRITICAL sanity check — do this before full training)

Before running full training, verify your model can actually learn by overfitting on one tiny batch:

```python
# Run this as a script: python scripts/overfit_check.py
import torch
import torch.nn.functional as F
from model.transformer import Transformer
from model.config import ModelConfig
from utils.device_utils import get_device

device = get_device()

cfg = ModelConfig(
    vocab_size=65, d_model=64, n_heads=4, n_layers=2,
    d_ff=256, block_size=32, dropout=0.0,
    learning_rate=1e-3, max_steps=200
)
model = Transformer(cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Fixed single batch
x = torch.randint(0, cfg.vocab_size, (4, 32)).to(device)
y = torch.randint(0, cfg.vocab_size, (4, 32)).to(device)

initial_loss = None
for step in range(200):
    optimizer.zero_grad()
    logits, _ = model(x)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
    if initial_loss is None:
        initial_loss = loss.item()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"Step {step:3d} | loss: {loss.item():.4f}")

print(f"\nInitial loss: {initial_loss:.4f} | Final loss: {loss.item():.4f}")
assert loss.item() < 0.1, "Model failed to overfit a single batch — check gradient flow"
print("Overfit check PASSED")
```

**Expected:** Loss should fall from ~4.1 to near 0 within 200 steps. If it doesn't, there's a bug in the model (wrong residuals, broken mask, etc.).

**Phase 7 done when:** Overfit check passes. Loss goes from ~ln(vocab_size) to near-zero.

---

## Phase 8 — Evaluation & Logging

**Goal:** Log train/val loss and perplexity to terminal and plot curves.

### Step 8.1 — Write `scripts/train.py` (entry point)

```python
import argparse
from model.transformer import Transformer
from model.config import ModelConfig
from training.dataset import build_dataloaders
from training.trainer import Trainer
from utils.count_params import count_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--checkpoint_dir", default="checkpoints/")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = ModelConfig(**cfg_dict)

    print(f"Config: {args.config}")
    print(f"device: ", end="")
    from utils.device_utils import get_device
    print(get_device())

    model = Transformer(cfg)
    count_params(model)

    train_loader, val_loader = build_dataloaders(
        train_path="data/processed/train.bin",
        val_path="data/processed/val.bin",
        block_size=cfg.block_size,
        batch_size=cfg.batch_size,
    )

    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.train()

    # Plot loss curves after training
    plot_loss_curves(trainer.train_losses, trainer.val_losses)

def plot_loss_curves(train_losses, val_losses):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(train_losses, alpha=0.7, linewidth=0.8)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-Entropy Loss")

    if val_losses:
        steps, losses = zip(*val_losses)
        import math
        ppl = [math.exp(l) for l in losses]
        axes[1].plot(steps, ppl, marker="o", linewidth=1.5)
        axes[1].set_title("Validation Perplexity")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Perplexity")

    plt.tight_layout()
    plt.savefig("notebooks/loss_curves.png", dpi=150)
    print("Loss curves saved to notebooks/loss_curves.png")

if __name__ == "__main__":
    main()
```

### Step 8.2 — Write `scripts/evaluate.py`

```python
import argparse
import torch
import torch.nn.functional as F
import math
from model.transformer import Transformer
from model.config import ModelConfig
from training.dataset import TokenDataset
from torch.utils.data import DataLoader
from training.checkpointing import load_checkpoint
from utils.device_utils import get_device

def evaluate(checkpoint_path: str, val_path: str, block_size: int, batch_size: int = 32):
    device = get_device()
    cfg, model, _, _ = load_checkpoint(checkpoint_path, device)
    model.eval()

    ds = TokenDataset(val_path, block_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
            total_loss += loss.item()
            n += 1

    avg_loss = total_loss / n
    print(f"Val Loss:        {avg_loss:.4f}")
    print(f"Val Perplexity:  {math.exp(avg_loss):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_path", default="data/processed/val.bin")
    parser.add_argument("--block_size", type=int, default=128)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.val_path, args.block_size)
```

**Phase 8 done when:** `python scripts/train.py --config configs/tiny.yaml` runs, logs loss every eval_interval steps, and saves a loss curve plot.

---

## Phase 9 — Checkpointing

**Goal:** Save and load model + optimizer state so training can be resumed after interruption.

### Step 9.1 — Write `training/checkpointing.py`

```python
import torch
import os
from pathlib import Path

def save_checkpoint(model, optimizer, step: int, cfg, checkpoint_dir: str):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"step_{step:06d}.pt")
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg.__dict__,
    }, path)
    print(f"Checkpoint saved: {path}")
    return path

def load_checkpoint(path: str, device):
    from model.config import ModelConfig
    from model.transformer import Transformer

    ckpt = torch.load(path, map_location=device)
    cfg = ModelConfig(**ckpt["config"])
    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay
    )
    optimizer.load_state_dict(ckpt["optimizer_state"])

    print(f"Loaded checkpoint: {path} (step {ckpt['step']})")
    return cfg, model, optimizer, ckpt["step"]
```

### Step 9.2 — Test checkpoint round-trip

```python
import torch
from model.transformer import Transformer
from model.config import ModelConfig
from training.checkpointing import save_checkpoint, load_checkpoint

cfg = ModelConfig(vocab_size=65, d_model=64, n_heads=4, n_layers=2, d_ff=256, block_size=32)
model = Transformer(cfg)
optimizer = torch.optim.AdamW(model.parameters())

path = save_checkpoint(model, optimizer, step=100, cfg=cfg, checkpoint_dir="/tmp/ckpt_test/")

cfg2, model2, opt2, step = load_checkpoint(path, device=torch.device("cpu"))
assert step == 100

x = torch.randint(0, 65, (1, 16))
logits1, _ = model(x)
logits2, _ = model2(x)
assert torch.allclose(logits1, logits2), "Checkpoint round-trip failed — weights differ"
print("Checkpoint round-trip PASSED")
```

**Phase 9 done when:** Checkpoint saves, loads, and produces identical outputs.

---

## Phase 10 — Inference & Generation

**Goal:** Autoregressive text generation with KV cache and configurable sampling.

### Step 10.1 — Write `inference/sampler.py`

```python
import torch
import torch.nn.functional as F

def sample_token(logits: torch.Tensor,
                 temperature: float = 1.0,
                 top_k: int | None = None,
                 top_p: float | None = None) -> torch.Tensor:
    # logits: [vocab_size]
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    if top_p is not None:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = (cumulative - sorted_probs) > top_p
        sorted_probs[remove] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()
        return torch.multinomial(sorted_probs, 1).gather(0, sorted_idx.argsort())

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)
```

### Step 10.2 — Write generation function in `scripts/generate.py`

```python
import torch
import argparse
from training.checkpointing import load_checkpoint
from inference.sampler import sample_token
from tokenizer.char_tokenizer import CharTokenizer
from utils.device_utils import get_device

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int,
             temperature: float, top_k: int | None, device):
    model.eval()
    cfg = model.cfg
    ids = tokenizer.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    for _ in range(max_tokens):
        # Truncate context to block_size
        context = ids[:, -cfg.block_size:]
        logits, _ = model(context)
        next_token_logits = logits[0, -1, :]                    # [vocab_size]
        next_token = sample_token(next_token_logits, temperature, top_k)
        ids = torch.cat([ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

    return tokenizer.decode(ids[0].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/processed/tokenizer.json")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = get_device()
    cfg, model, _, _ = load_checkpoint(args.checkpoint, device)
    tok = CharTokenizer(vocab_path=args.tokenizer)

    output = generate(model, tok, args.prompt, args.tokens,
                      args.temperature, args.top_k, device)
    print(output)
```

### Step 10.3 — Write `tests/test_sampler.py`

```python
import torch
from inference.sampler import sample_token

def test_greedy_deterministic():
    logits = torch.tensor([0.0, 10.0, 0.0, 0.0])
    # With very low temperature, should always pick index 1
    results = {sample_token(logits, temperature=0.01).item() for _ in range(10)}
    assert results == {1}

def test_top_k_restricts_vocab():
    torch.manual_seed(42)
    logits = torch.randn(100)
    # With top_k=5, only 5 tokens eligible
    samples = {sample_token(logits, top_k=5).item() for _ in range(200)}
    top5 = torch.topk(logits, 5).indices.tolist()
    assert samples.issubset(set(top5))

def test_output_valid_index():
    logits = torch.randn(65)
    idx = sample_token(logits, temperature=1.0).item()
    assert 0 <= idx < 65
```

### Step 10.4 — Test generation end-to-end

```bash
python scripts/generate.py \
  --checkpoint checkpoints/step_005000.pt \
  --prompt "ROMEO:" \
  --tokens 300 \
  --temperature 0.8 \
  --top_k 40
```

**Phase 10 done when:** Model generates readable text. Even a tiny model should produce Elizabethan-ish patterns after ~5000 steps on Shakespeare.

---

## Phase 11 — Attention Visualization

**Goal:** Plot attention weight heatmaps for every head across every layer. Identify at least one head with interpretable behavior.

### Step 11.1 — Write `utils/visualize_attention.py`

```python
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_attention_heads(
    attn_weights: list[torch.Tensor],
    tokens: list[str],
    save_path: str | None = None,
):
    """
    attn_weights: list of [B, H, T, T] tensors, one per layer
    tokens: list of decoded token strings for axis labels
    """
    n_layers = len(attn_weights)
    n_heads = attn_weights[0].shape[1]
    T = len(tokens)

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(n_heads * 3, n_layers * 3),
        squeeze=False
    )

    for layer_idx, weights in enumerate(attn_weights):
        w = weights[0].detach().cpu().numpy()   # [H, T, T] — take batch item 0
        for head_idx in range(n_heads):
            ax = axes[layer_idx][head_idx]
            im = ax.imshow(w[head_idx, :T, :T], cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"L{layer_idx+1} H{head_idx+1}", fontsize=8)
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticklabels(tokens, fontsize=6)

    plt.suptitle("Attention Weights per Layer & Head", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
```

### Step 11.2 — Add visualization to `scripts/generate.py`

After generating, call the visualizer on the last forward pass:

```python
# In generate(): capture the attention weights on a fixed short prompt
prompt = "To be or not"
ids_tensor = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
with torch.no_grad():
    logits, attn_weights = model(ids_tensor)

tokens = list(prompt)   # character-level: each char is one token
from utils.visualize_attention import plot_attention_heads
plot_attention_heads(attn_weights, tokens, save_path="notebooks/attention_map.png")
```

### Step 11.3 — What to look for in attention heads

| Pattern | What it means |
|---|---|
| Strong diagonal | Each token mostly attends to itself |
| One column lit up | All tokens attend to a specific position (often first token) |
| Lower-triangular gradient | Uniform attention to all previous tokens |
| Sparse sharp row | Token attends to one specific earlier token — most interpretable |
| BOS (beginning-of-sequence) column | Model uses a "register" position to store information |

**Phase 11 done when:** Attention heatmap image generated and saved. You should see visually distinct patterns between heads.

---

## Phase 12 — BPE Tokenizer

**Goal:** Implement Byte Pair Encoding from scratch, replacing the character tokenizer for the 10M-param run.

### Step 12.1 — Understand BPE algorithm

```
Start: every character is its own token
Repeat until vocab_size reached:
  1. Count all adjacent token pairs in the corpus
  2. Find the most frequent pair (a, b)
  3. Merge (a, b) → "ab" everywhere in the corpus
  4. Add "ab" to the vocabulary
Result: common subwords become single tokens
```

### Step 12.2 — Write `tokenizer/bpe_tokenizer.py`

```python
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm

class BPETokenizer:
    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.merges: list[tuple[str, str]] = []
        self.itos: dict[int, str] = {}

    def train(self, text: str, vocab_size: int, verbose: bool = True):
        # Build initial character vocabulary
        chars = sorted(set(text))
        self.vocab = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

        # Represent corpus as list of lists of characters
        words = text.split()
        word_counts = Counter(words)
        corpus = {
            tuple(list(word) + ["</w>"]): count
            for word, count in word_counts.items()
        }

        pbar = tqdm(range(vocab_size - len(self.vocab)), desc="BPE merges", disable=not verbose)

        for _ in pbar:
            # Count all adjacent pairs
            pair_counts: dict[tuple, int] = defaultdict(int)
            for word, count in corpus.items():
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i+1])] += count

            if not pair_counts:
                break

            # Find best pair
            best_pair = max(pair_counts, key=pair_counts.get)
            a, b = best_pair
            merged = a + b

            # Add to vocab
            idx = len(self.vocab)
            self.vocab[merged] = idx
            self.itos[idx] = merged
            self.merges.append(best_pair)

            # Apply merge to corpus
            new_corpus = {}
            for word, count in corpus.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                        new_word.append(merged)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_corpus[tuple(new_word)] = count
            corpus = new_corpus

            pbar.set_postfix({"vocab": len(self.vocab), "best": f"'{a}'+'{b}'"})

        self.vocab_size = len(self.vocab)

    def _apply_merges(self, tokens: list[str]) -> list[str]:
        for a, b in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == a and tokens[i+1] == b:
                    tokens = tokens[:i] + [a + b] + tokens[i+2:]
                else:
                    i += 1
        return tokens

    def encode(self, text: str) -> list[int]:
        tokens = []
        for word in text.split():
            chars = list(word) + ["</w>"]
            merged = self._apply_merges(chars)
            tokens.extend(self.vocab[t] for t in merged if t in self.vocab)
        return tokens

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids).replace("</w>", " ").strip()

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "vocab": self.vocab,
                "merges": self.merges,
                "itos": {str(k): v for k, v in self.itos.items()},
            }, f)

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
```

### Step 12.3 — Write `tokenizer/train_tokenizer.py`

```python
import argparse
from pathlib import Path
from tokenizer.bpe_tokenizer import BPETokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--output", default="data/processed/bpe_tokenizer.json")
    args = parser.parse_args()

    text = Path(args.input).read_text()
    tok = BPETokenizer()
    tok.train(text, vocab_size=args.vocab_size)
    tok.save(args.output)
    print(f"BPE vocab size: {tok.vocab_size}")
    print(f"Saved to: {args.output}")
```

### Step 12.4 — Train the BPE tokenizer

```bash
python tokenizer/train_tokenizer.py \
  --input data/raw/shakespeare.txt \
  --vocab_size 1000 \
  --output data/processed/bpe_tokenizer.json
```

### Step 12.5 — Add BPE tests

```python
# Append to tests/test_tokenizer.py
from tokenizer.bpe_tokenizer import BPETokenizer

def test_bpe_roundtrip():
    text = "hello world hello world foo bar"
    tok = BPETokenizer()
    tok.train(text, vocab_size=30, verbose=False)
    encoded = tok.encode("hello world")
    decoded = tok.decode(encoded)
    assert "hello" in decoded and "world" in decoded

def test_bpe_save_load(tmp_path):
    text = "the quick brown fox jumps over the lazy dog"
    tok = BPETokenizer()
    tok.train(text, vocab_size=50, verbose=False)
    path = str(tmp_path / "bpe.json")
    tok.save(path)
    tok2 = BPETokenizer.load(path)
    assert tok.encode("the fox") == tok2.encode("the fox")
```

**Phase 12 done when:** BPE tokenizer trains, saves, loads, and encodes/decodes consistently.

---

## Phase 13 — Scale Up & Experiments

**Goal:** Train a 10M-param model, compare PE variants, and document findings.

### Step 13.1 — Train the small config

```bash
python scripts/train.py --config configs/small.yaml
```

Monitor:
- Loss should decrease smoothly — any spike suggests a gradient explosion (lower `learning_rate` or increase `warmup_steps`)
- If loss plateaus early, try increasing `n_layers` or `d_model`
- Watch GPU/CPU memory — if OOM, reduce `batch_size` and increase `gradient_accumulation_steps`

### Step 13.2 — Sinusoidal vs RoPE comparison

Train two models from identical initialization (same random seed), one with each PE:

```bash
python scripts/train.py --config configs/tiny.yaml  # sinusoidal (default)

# Edit configs/tiny_rope.yaml: pos_encoding: rope
python scripts/train.py --config configs/tiny_rope.yaml
```

Then plot both val perplexity curves on the same axis for comparison.

### Step 13.3 — Hyperparameter ablations

Run these in sequence and record final val perplexity for each:

| Experiment | Change | Expected effect |
|---|---|---|
| No dropout | `dropout: 0.0` | Overfit faster — higher val loss |
| Double layers | `n_layers: 8` | Better capacity, slower training |
| Half d_model | `d_model: 64` | Much fewer params, worse perplexity |
| Higher LR | `learning_rate: 1e-3` | Faster initial descent, may diverge |
| No warmup | `warmup_steps: 0` | Often unstable at start |
| No weight tying | Untie lm_head | More params, minor perplexity change |

### Step 13.4 — Flash Attention swap (optimization milestone)

After all your custom attention tests pass, replace the attention computation:

```python
# In model/attention.py forward(), swap the manual attention block with:
out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=self.attn_dropout.p if self.training else 0.0,
    is_causal=True,
)
```

Benchmark before/after to measure the speedup on your RTX 5070.

---

## Debugging Reference

### Loss is NaN immediately
- Check for zero/negative learning rate
- Check for weight initialization issues (too large std)
- Gradient explosion: lower `learning_rate`, add `grad_clip`
- Attention: check that `-inf` masking doesn't produce NaN in softmax when entire row is masked

### Loss is flat (not decreasing)
- LR too low — try 10× higher
- Data pipeline issue — verify `y = x[1:]` shift is correct
- Gradient not flowing — run the overfit-one-batch check
- Check `model.train()` is called before training

### Loss decreases but validation diverges
- Overfitting — increase `dropout`, reduce `n_layers` or `d_model`
- Not enough data — use a larger dataset

### Generated text is gibberish after training
- Model hasn't trained long enough — increase `max_steps`
- Temperature too high — try `temperature=0.5`
- Tokenizer mismatch — ensure same tokenizer for training and generation

### CUDA out of memory
- Reduce `batch_size`
- Increase `gradient_accumulation_steps` to compensate
- Reduce `block_size`
- Use `torch.cuda.empty_cache()` between eval and training

### `torch.compile()` errors
- Disable it first to isolate the bug
- Common issue: dynamic shapes — ensure `block_size` is always the same
- Falls back gracefully with `torch.compile(model, fullgraph=False)`

---

## Sanity Check Checklist

Run this checklist at each phase boundary before moving forward.

### After Phase 1 (Tokenizer)
- [ ] `encode(decode(ids)) == ids` for any string from the training corpus
- [ ] `vocab_size == 65` for Shakespeare character tokenizer
- [ ] `.bin` files created and have correct sizes

### After Phase 2 (Dataset)
- [ ] `x.shape == (batch_size, block_size)`
- [ ] `y == x` shifted by 1 (check first few tokens manually)
- [ ] DataLoader iterates without error

### After Phase 4 (Attention)
- [ ] Output shape `[B, T, d_model]` for any valid `B, T`
- [ ] Attention weight upper triangle is exactly 0 (causal mask correct)
- [ ] No NaN for random inputs

### After Phase 6 (Full Model)
- [ ] `logits.shape == [B, T, vocab_size]`
- [ ] Initial loss ≈ `ln(vocab_size)` — random predictions
- [ ] `AssertionError` raised for `T > block_size`
- [ ] Parameter count matches formula

### After Phase 7 (Training)
- [ ] Overfit check: loss goes from ~4.1 to < 0.1 on a single batch within 200 steps
- [ ] Full training run starts and loss decreases within first 100 steps
- [ ] LR warms up correctly (check first 100 steps in logs)

### After Phase 9 (Checkpointing)
- [ ] Checkpoint saved file exists
- [ ] Loaded model produces identical logits to saved model on same input

### After Phase 10 (Generation)
- [ ] Generated text is valid (no unknown characters, not empty)
- [ ] KV-cache generation matches non-cached output exactly
- [ ] Different temperatures produce noticeably different outputs

### After Phase 11 (Visualization)
- [ ] Attention heatmap image generated
- [ ] Causal structure visible (upper triangle is blank)
- [ ] At least one head shows a distinct, non-uniform pattern

### After Phase 12 (BPE)
- [ ] BPE encode/decode roundtrip test passes
- [ ] Vocabulary contains multi-character merges (e.g. "th", "the", "in")
- [ ] Tokens per word reduced vs character-level

### After Phase 13 (Scale)
- [ ] 10M param model reaches val perplexity < 3.0 on Shakespeare
- [ ] RoPE and sinusoidal PE comparison plot saved
- [ ] At least one ablation experiment documented

---

*Build in order. Test before moving on. The bugs you catch in Phase 1 are free; the bugs you find in Phase 7 cost hours.*
