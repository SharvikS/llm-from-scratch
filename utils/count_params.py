def count_params(model) -> int:
    """Print a per-module parameter breakdown and return the trainable count.

    Tied weights (e.g. lm_head sharing the embedding) are counted once by
    PyTorch's parameter iterator.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{'Parameter Group':<30} {'Count':>15}")
    print("-" * 47)
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:<28} {n:>15,}")
    print("-" * 47)
    print(f"{'Total':<30} {total:>15,}")
    print(f"{'Trainable':<30} {trainable:>15,}")
    return trainable
