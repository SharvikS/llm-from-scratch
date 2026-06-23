import pytest
import torch

from model.attention import MultiHeadSelfAttention


@pytest.fixture
def attn():
    return MultiHeadSelfAttention(d_model=64, n_heads=4, block_size=32, dropout=0.0)


def test_output_shape(attn):
    x = torch.randn(2, 16, 64)
    out, weights = attn(x)
    assert out.shape == (2, 16, 64)


def test_attention_weight_shape(attn):
    x = torch.randn(2, 16, 64)
    _, weights = attn(x)
    assert weights.shape == (2, 4, 16, 16)   # [B, H, T, T]


def test_causal_mask(attn):
    # No query position may attend to a strictly future key position.
    x = torch.randn(1, 8, 64)
    _, weights = attn(x)
    weights = weights.squeeze(0)             # [H, T, T]
    for h in range(weights.shape[0]):
        for t in range(8):
            future = weights[h, t, t + 1:]
            assert future.sum().item() < 1e-6, \
                f"Head {h}, position {t} attends to a future token"


def test_no_nan(attn):
    x = torch.randn(2, 16, 64)
    out, _ = attn(x)
    assert not torch.isnan(out).any()


def test_rope_variant():
    rope_attn = MultiHeadSelfAttention(
        d_model=64, n_heads=4, block_size=32, dropout=0.0, pos_encoding="rope"
    )
    x = torch.randn(2, 16, 64)
    out, _ = rope_attn(x)
    assert out.shape == (2, 16, 64)
    assert not torch.isnan(out).any()


def test_kv_cache_matches_full_pass():
    # Feeding tokens one at a time with a KV cache must match a single full pass.
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(d_model=32, n_heads=4, block_size=16, dropout=0.0)
    attn.eval()
    x = torch.randn(1, 6, 32)

    with torch.no_grad():
        full_out, _ = attn(x)

        cached = []
        past = None
        for t in range(x.size(1)):
            step_out, _, past = attn(
                x[:, t : t + 1], past_kv=past, use_cache=True, input_pos=t
            )
            cached.append(step_out)
        cached_out = torch.cat(cached, dim=1)

    assert torch.allclose(full_out, cached_out, atol=1e-5)
