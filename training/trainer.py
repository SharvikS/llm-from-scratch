import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.config import ModelConfig
from model.transformer import Transformer
from training.checkpointing import save_checkpoint
from training.lr_scheduler import get_lr
from utils.device_utils import get_device


class Trainer:
    """Training loop with gradient accumulation, cosine LR, grad clipping,
    periodic evaluation, and checkpointing."""

    def __init__(self, model: Transformer, cfg: ModelConfig,
                 train_loader: DataLoader, val_loader: DataLoader,
                 checkpoint_dir: str = "checkpoints/"):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.device = get_device()
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )

        self.step = 0
        self.train_losses = []
        self.val_losses = []

    def _loss(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        logits, _ = self.model(x)
        return F.cross_entropy(logits.view(-1, self.cfg.vocab_size), y.view(-1))

    @torch.no_grad()
    def evaluate(self, n_batches: int = 20) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for x, y in self.val_loader:
            if count >= n_batches:
                break
            total += self._loss(x, y).item()
            count += 1
        self.model.train()
        return total / max(count, 1)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.cfg.max_steps, desc="Training")

        while self.step < self.cfg.max_steps:
            lr = get_lr(self.step, self.cfg.warmup_steps, self.cfg.max_steps,
                        self.cfg.learning_rate, self.cfg.min_lr)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            step_loss = 0.0
            for _ in range(self.cfg.gradient_accumulation_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)
                loss = self._loss(x, y) / self.cfg.gradient_accumulation_steps
                loss.backward()
                step_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step += 1
            self.train_losses.append(step_loss)

            pbar.update(1)
            pbar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{lr:.2e}"})

            if self.step % self.cfg.eval_interval == 0:
                val_loss = self.evaluate()
                self.val_losses.append((self.step, val_loss))
                ppl = torch.exp(torch.tensor(val_loss)).item()
                tqdm.write(f"Step {self.step:5d} | train {step_loss:.4f} "
                           f"| val {val_loss:.4f} | ppl {ppl:.2f}")

            if self.step % self.cfg.save_interval == 0:
                save_checkpoint(self.model, self.optimizer, self.step,
                                self.cfg, self.checkpoint_dir)

        pbar.close()
        # Always save a final checkpoint at the last step.
        save_checkpoint(self.model, self.optimizer, self.step,
                        self.cfg, self.checkpoint_dir)
        print("Training complete.")
