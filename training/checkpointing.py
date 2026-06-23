import dataclasses
import os
from pathlib import Path

import torch

from model.config import ModelConfig
from model.transformer import Transformer


def _config_to_dict(cfg: ModelConfig) -> dict:
    """Serialize only declared dataclass fields (drops derived attrs like d_k)."""
    names = {f.name for f in dataclasses.fields(ModelConfig)}
    return {k: v for k, v in cfg.__dict__.items() if k in names}


def save_checkpoint(model, optimizer, step: int, cfg: ModelConfig, checkpoint_dir: str):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"step_{step:06d}.pt")
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": _config_to_dict(cfg),
        },
        path,
    )
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(path: str, device):
    """Rebuild model + optimizer from a checkpoint. Returns (cfg, model, opt, step)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])

    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )
    optimizer.load_state_dict(ckpt["optimizer_state"])

    print(f"Loaded checkpoint: {path} (step {ckpt['step']})")
    return cfg, model, optimizer, ckpt["step"]
