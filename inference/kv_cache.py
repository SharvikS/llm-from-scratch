"""KV-cache utilities for fast autoregressive generation.

Without a cache, each generation step recomputes attention over the whole
prefix -> O(T^2) total work. With a cache we store each layer's K and V and
only run the new token through the model -> O(T) total.

The actual cache here is just the per-layer list of (K, V) tensors threaded
through `Transformer.step`; this module documents the contract and offers a
small helper to drive a full cached generation loop.
"""
import torch

from inference.sampler import sample_token


@torch.no_grad()
def generate_cached(model, idx: torch.Tensor, max_new_tokens: int,
                    temperature: float = 1.0, top_k: int | None = None,
                    top_p: float | None = None) -> torch.Tensor:
    """Autoregressively extend `idx` [1, T] using per-layer KV caches.

    Prefills the prompt in a single step, then decodes one token at a time.
    Total generated length must stay within the model's block_size, since the
    positional tables and causal mask are sized to it.
    """
    model.eval()
    block_size = model.cfg.block_size
    assert idx.size(1) + max_new_tokens <= block_size, (
        f"prompt ({idx.size(1)}) + new ({max_new_tokens}) exceeds "
        f"block_size ({block_size}); cached decode does not slide the window"
    )

    # Prefill: run the whole prompt once to populate the caches.
    logits, caches = model.step(idx, caches=None, input_pos=0)
    next_logits = logits[0, -1, :]
    next_token = sample_token(next_logits, temperature, top_k, top_p)
    idx = torch.cat([idx, next_token.view(1, 1)], dim=1)

    pos = idx.size(1) - 1  # absolute position of the next-to-feed token
    for _ in range(max_new_tokens - 1):
        logits, caches = model.step(idx[:, -1:], caches=caches, input_pos=pos)
        next_token = sample_token(logits[0, -1, :], temperature, top_k, top_p)
        idx = torch.cat([idx, next_token.view(1, 1)], dim=1)
        pos += 1

    return idx
