import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.positional_encoding import apply_rope, precompute_rope_freqs


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention, written from scratch.

    Default training path: `out, weights = attn(x)`.

    Optional incremental decoding: pass `use_cache=True` (and a `past_kv` tuple
    after the first step) to get `out, weights, present_kv` back, where
    present_kv is the concatenated (K, V) to feed into the next step. The
    `input_pos` argument is the absolute position of the first query token,
    used both for RoPE rotation and to slice the causal mask.
    """

    def __init__(self, d_model: int, n_heads: int, block_size: int,
                 dropout: float = 0.1, pos_encoding: str = "sinusoidal"):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.pos_encoding = pos_encoding

        # Fused Q, K, V projection for efficiency, then a separate output proj.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask over absolute positions: True = "do not attend".
        mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

        if pos_encoding == "rope":
            cos, sin = precompute_rope_freqs(self.d_k, max_len=block_size)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

    def forward(self, x, past_kv=None, use_cache: bool = False, input_pos: int = 0):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)                            # [B, T, 3*d_model]
        q, k, v = qkv.split(C, dim=-1)                    # each [B, T, d_model]

        # -> [B, H, T, d_k]
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # RoPE rotates queries and keys (not values) by their absolute position.
        if self.pos_encoding == "rope":
            q = apply_rope(q, self.rope_cos, self.rope_sin, offset=input_pos)
            k = apply_rope(k, self.rope_cos, self.rope_sin, offset=input_pos)

        # Prepend cached keys/values for incremental decoding.
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present_kv = (k, v) if use_cache else None

        Tk = k.size(2)
        scale = math.sqrt(self.d_k)
        scores = (q @ k.transpose(-2, -1)) / scale        # [B, H, T, Tk]

        # Query row i is absolute position (input_pos + i); slice the mask so it
        # works for both full training passes and single-token cached steps.
        mask = self.causal_mask[input_pos : input_pos + T, :Tk]
        scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)          # [B, H, T, Tk]
        attn_weights = self.attn_dropout(attn_weights)

        out = attn_weights @ v                            # [B, H, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # [B, T, d_model]
        out = self.resid_dropout(self.out_proj(out))

        if use_cache:
            return out, attn_weights, present_kv
        return out, attn_weights
