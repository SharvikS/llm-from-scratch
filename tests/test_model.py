import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model.config import ModelConfig
from model.transformer import Transformer
from training.dataset import TokenDataset


@pytest.fixture
def cfg():
    return ModelConfig(
        vocab_size=65, d_model=64, n_heads=4, n_layers=2,
        d_ff=256, block_size=32, dropout=0.0,
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
    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    logits, _ = model(idx)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), idx.view(-1))
    expected = torch.log(torch.tensor(float(cfg.vocab_size)))  # ~4.17 for 65
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


def test_param_count_positive(model):
    assert model.num_params() > 0


def test_cached_step_matches_full_forward(cfg):
    # Greedy logits from incremental KV-cache decoding must match a full pass.
    torch.manual_seed(0)
    model = Transformer(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 10))

    with torch.no_grad():
        full_logits, _ = model(idx)

        caches = None
        step_logits = []
        for t in range(idx.size(1)):
            lg, caches = model.step(idx[:, t : t + 1], caches, input_pos=t)
            step_logits.append(lg)
        cached_logits = torch.cat(step_logits, dim=1)

    assert torch.allclose(full_logits, cached_logits, atol=1e-4)


def test_dataset_shapes(tmp_path):
    data = np.arange(1000, dtype=np.uint16)
    bin_file = str(tmp_path / "test.bin")
    data.tofile(bin_file)

    ds = TokenDataset(bin_file, block_size=64)
    x, y = ds[0]
    assert x.shape == (64,)
    assert y.shape == (64,)
    # y is x shifted by one token
    assert torch.equal(y[:-1], x[1:])
