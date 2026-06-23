import math


def get_lr(step: int, warmup_steps: int, max_steps: int,
           learning_rate: float, min_lr: float) -> float:
    """Linear warmup then cosine decay to min_lr.

    step 0..warmup_steps:      linear ramp 0 -> learning_rate
    warmup_steps..max_steps:   cosine decay learning_rate -> min_lr
    beyond max_steps:          held at min_lr
    """
    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, (max_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (learning_rate - min_lr)
