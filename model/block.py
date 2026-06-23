import torch.nn as nn

from model.attention import MultiHeadSelfAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: x + Attn(LN(x)), then x + FFN(LN(x)).

    Pre-norm (LayerNorm before each sub-layer) is the GPT-2+ formulation and is
    more stable to train than the original post-norm design.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 block_size: int, dropout: float, pos_encoding: str):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, block_size, dropout, pos_encoding)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, past_kv=None, use_cache: bool = False, input_pos: int = 0):
        if use_cache:
            attn_out, attn_weights, present_kv = self.attn(
                self.ln1(x), past_kv=past_kv, use_cache=True, input_pos=input_pos
            )
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
            return x, attn_weights, present_kv

        attn_out, attn_weights = self.attn(self.ln1(x), input_pos=input_pos)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, attn_weights
