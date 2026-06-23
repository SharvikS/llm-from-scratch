import torch
import torch.nn as nn

from model.block import TransformerBlock
from model.config import ModelConfig
from model.layer_norm import LayerNorm
from model.positional_encoding import SinusoidalPE


class Transformer(nn.Module):
    """Decoder-only (GPT-style) Transformer language model.

    token embedding -> positional encoding -> N pre-norm blocks -> final
    LayerNorm -> tied linear head producing logits over the vocabulary.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Sinusoidal PE adds to the embeddings; RoPE is applied inside attention.
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

        # Weight tying: share the embedding and output projection (GPT-2 style).
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

    def _embed(self, idx: torch.Tensor, offset: int = 0) -> torch.Tensor:
        x = self.token_embedding(idx)
        if self.pos_embedding is not None:
            x = self.pos_embedding(x, offset=offset)
        else:
            x = self.embed_dropout(x)
        return x

    def forward(self, idx: torch.Tensor):
        """Full (training / scoring) forward pass over a [B, T] token batch."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"Sequence length {T} > block_size {self.cfg.block_size}"

        x = self._embed(idx)
        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block(x)
            all_attn_weights.append(attn_weights)

        x = self.ln_final(x)
        logits = self.lm_head(x)            # [B, T, vocab_size]
        return logits, all_attn_weights

    @torch.no_grad()
    def step(self, idx: torch.Tensor, caches, input_pos: int):
        """One incremental decode step using per-layer KV caches.

        idx:       [B, T_new] new token(s) — usually T_new == 1
        caches:    list of (K, V) per layer, or None on the first step
        input_pos: absolute position of the first new token
        Returns (logits, new_caches).
        """
        x = self._embed(idx, offset=input_pos)
        new_caches = []
        for i, block in enumerate(self.blocks):
            past = caches[i] if caches is not None else None
            x, _, present = block(x, past_kv=past, use_cache=True, input_pos=input_pos)
            new_caches.append(present)

        x = self.ln_final(x)
        logits = self.lm_head(x)            # [B, T_new, vocab_size]
        return logits, new_caches

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
