import math

import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding from "Attention Is All You Need".

    Adds a deterministic position signal to token embeddings. The buffer is
    registered (not a parameter) so it is not trained but still moves with the
    module across devices and is saved in the state dict.
    """

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                       # [T, d_model]
        pos = torch.arange(max_len).unsqueeze(1).float()         # [T, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                        # [d_model/2]
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))              # [1, T, d_model]

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: [B, T, d_model]; offset lets cached generation index absolute pos.
        T = x.size(1)
        x = x + self.pe[:, offset : offset + T]
        return self.dropout(x)


def precompute_rope_freqs(d_k: int, max_len: int = 4096):
    """Precompute the cos/sin rotation tables for RoPE.

    Returns two [max_len, d_k/2] tensors. Position information is encoded by
    rotating Q and K, so the attention dot product depends on relative offset.
    """
    theta = 1.0 / (10000 ** (torch.arange(0, d_k, 2).float() / d_k))  # [d_k/2]
    pos = torch.arange(max_len).float()                               # [T]
    freqs = torch.outer(pos, theta)                                   # [T, d_k/2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0):
    """Rotate the per-head vectors x by their positional angle.

    x:        [B, H, T, d_k]
    cos, sin: [max_len, d_k/2]
    offset:   absolute position of x[..., 0, :] (non-zero during cached decode)
    """
    T = x.size(2)
    d_k_half = cos.size(1)
    cos_t = cos[offset : offset + T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, d_k/2]
    sin_t = sin[offset : offset + T].unsqueeze(0).unsqueeze(0)
    x1 = x[..., :d_k_half]
    x2 = x[..., d_k_half:]
    return torch.cat([x1 * cos_t - x2 * sin_t, x2 * cos_t + x1 * sin_t], dim=-1)
