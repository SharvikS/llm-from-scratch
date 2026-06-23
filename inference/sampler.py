import torch
import torch.nn.functional as F


def sample_token(logits: torch.Tensor,
                 temperature: float = 1.0,
                 top_k: int | None = None,
                 top_p: float | None = None) -> torch.Tensor:
    """Sample one token id from a 1-D logits vector [vocab_size].

    temperature: <1 sharpens, >1 flattens, ->0 approaches greedy/argmax.
    top_k:       keep only the k highest-logit tokens before sampling.
    top_p:       keep the smallest set of tokens whose cumulative prob >= top_p
                 (nucleus sampling). top_k and top_p may be combined.
    Returns a LongTensor of shape [1].
    """
    logits = logits / max(temperature, 1e-6)

    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth_value = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < kth_value, float("-inf"))

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        # Drop tokens once the running mass (excluding the current one) exceeds
        # top_p; the first token is always kept.
        remove = (cumprobs - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            -1, sorted_idx, sorted_logits
        )

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
