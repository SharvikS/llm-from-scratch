from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Single source of truth for architecture + training hyperparameters.

    Loaded from a YAML file (see configs/) via ModelConfig(**yaml_dict) or
    utils.config_loader.load_config.
    """

    # --- Architecture ---
    vocab_size: int = 65
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512            # typically 4 x d_model
    block_size: int = 128      # max sequence length / context window
    dropout: float = 0.1

    # --- Positional encoding ---
    pos_encoding: str = "sinusoidal"   # "sinusoidal" | "rope"

    # --- Training ---
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    max_steps: int = 5000
    eval_interval: int = 500
    save_interval: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    min_lr: float = 3e-5

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.pos_encoding in ("sinusoidal", "rope"), (
            f"pos_encoding must be 'sinusoidal' or 'rope', got {self.pos_encoding!r}"
        )
        # derived
        self.d_k = self.d_model // self.n_heads
