import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias.

    PyTorch's built-in nn.LayerNorm always has a bias; GPT-2-style models often
    drop it, so we expose `bias=False`. Normalizes over the feature dimension.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)
