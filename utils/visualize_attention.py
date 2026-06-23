import matplotlib.pyplot as plt


def plot_attention_heads(attn_weights, tokens, save_path: str | None = None):
    """Plot attention heatmaps for every (layer, head).

    attn_weights: list of [B, H, T, T] tensors, one per layer (as returned by
                  Transformer.forward). Batch item 0 is visualized.
    tokens:       decoded token strings used as axis tick labels.
    save_path:    if given, write a PNG; otherwise show interactively.
    """
    n_layers = len(attn_weights)
    n_heads = attn_weights[0].shape[1]
    T = len(tokens)

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(n_heads * 3, n_layers * 3),
        squeeze=False,
    )

    for layer_idx, weights in enumerate(attn_weights):
        w = weights[0].detach().cpu().numpy()   # [H, T, T]
        for head_idx in range(n_heads):
            ax = axes[layer_idx][head_idx]
            ax.imshow(w[head_idx, :T, :T], cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"L{layer_idx + 1} H{head_idx + 1}", fontsize=8)
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(tokens, rotation=90, fontsize=6)
            ax.set_yticklabels(tokens, fontsize=6)

    plt.suptitle("Attention Weights per Layer & Head", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
