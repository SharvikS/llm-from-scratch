import torch

from inference.sampler import sample_token


def test_greedy_deterministic():
    logits = torch.tensor([0.0, 10.0, 0.0, 0.0])
    # Near-zero temperature collapses to argmax (index 1).
    results = {sample_token(logits, temperature=0.01).item() for _ in range(10)}
    assert results == {1}


def test_top_k_restricts_vocab():
    torch.manual_seed(42)
    logits = torch.randn(100)
    samples = {sample_token(logits, top_k=5).item() for _ in range(200)}
    top5 = set(torch.topk(logits, 5).indices.tolist())
    assert samples.issubset(top5)


def test_top_p_restricts_vocab():
    torch.manual_seed(0)
    logits = torch.randn(50)
    # A small nucleus should only ever yield a few distinct tokens.
    samples = {sample_token(logits, top_p=0.5).item() for _ in range(300)}
    assert 1 <= len(samples) <= 10


def test_output_valid_index():
    logits = torch.randn(65)
    idx = sample_token(logits, temperature=1.0).item()
    assert 0 <= idx < 65
