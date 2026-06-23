"""FastAPI backend for the llm-from-scratch web UI.

Thin wrapper around the existing model / tokenizer / inference code. Nothing
here re-implements the transformer — it loads a checkpoint (or falls back to a
freshly-initialised "demo" model when none exists yet) and exposes:

    GET  /api/health          liveness + whether a real checkpoint is loaded
    GET  /api/model           loaded config, parameter count, device
    GET  /api/checkpoints     .pt files discovered under checkpoints/
    POST /api/load            load a specific checkpoint by name
    POST /api/tokenize        char-level encode of arbitrary text
    POST /api/attention       per-layer/head attention weights for a prompt
    POST /api/generate        non-streaming generation (returns full text)
    POST /api/generate/stream Server-Sent-Events token-by-token generation

Run from the repo root:

    pip install -r webui/backend/requirements.txt
    uvicorn webui.backend.server:app --reload --port 8000
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

# --- make the repo root importable so `model`, `tokenizer`, ... resolve ------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from inference.sampler import sample_token  # noqa: E402
from model.config import ModelConfig  # noqa: E402
from model.transformer import Transformer  # noqa: E402
from tokenizer.char_tokenizer import CharTokenizer  # noqa: E402
from training.checkpointing import load_checkpoint  # noqa: E402
from utils.config_loader import load_config  # noqa: E402
from utils.device_utils import get_device  # noqa: E402

CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
TOKENIZER_PATH = REPO_ROOT / "data" / "processed" / "tokenizer.json"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tiny.yaml"


# ----------------------------------------------------------------------------
# Model registry — a single process-wide loaded model + tokenizer.
# ----------------------------------------------------------------------------
class ModelState:
    def __init__(self) -> None:
        self.device = get_device()
        self.model: Optional[Transformer] = None
        self.cfg: Optional[ModelConfig] = None
        self.tokenizer: Optional[CharTokenizer] = None
        self.checkpoint: Optional[str] = None  # None => untrained demo model
        self._load_tokenizer()
        self._load_initial_model()

    # -- tokenizer ----------------------------------------------------------
    def _load_tokenizer(self) -> None:
        if TOKENIZER_PATH.exists():
            self.tokenizer = CharTokenizer(vocab_path=str(TOKENIZER_PATH))
        else:
            # Minimal printable-ASCII fallback so the UI still functions.
            sample = "".join(chr(c) for c in range(32, 127)) + "\n"
            self.tokenizer = CharTokenizer(text=sample)

    # -- model --------------------------------------------------------------
    def _build_demo_model(self) -> None:
        """Untrained model so the app is usable before any training run."""
        cfg = load_config(str(DEFAULT_CONFIG)) if DEFAULT_CONFIG.exists() else ModelConfig()
        cfg.vocab_size = self.tokenizer.vocab_size
        cfg.dropout = 0.0
        model = Transformer(cfg).to(self.device)
        model.eval()
        self.model, self.cfg, self.checkpoint = model, cfg, None

    def _load_initial_model(self) -> None:
        latest = self._latest_checkpoint()
        if latest is not None:
            try:
                self.load(latest.name)
                return
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[webui] failed to load {latest.name}: {exc}; using demo model")
        self._build_demo_model()

    def _latest_checkpoint(self) -> Optional[Path]:
        ckpts = sorted(CHECKPOINT_DIR.glob("*.pt"))
        return ckpts[-1] if ckpts else None

    def load(self, name: str) -> None:
        path = CHECKPOINT_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {name!r} not found")
        cfg, model, _, _ = load_checkpoint(str(path), self.device)
        model.eval()
        self.model, self.cfg, self.checkpoint = model, cfg, name

    # -- introspection ------------------------------------------------------
    def info(self) -> dict:
        assert self.model is not None and self.cfg is not None
        return {
            "checkpoint": self.checkpoint,
            "trained": self.checkpoint is not None,
            "device": str(self.device),
            "num_params": self.model.num_params(),
            "vocab_size": self.cfg.vocab_size,
            "config": {
                "d_model": self.cfg.d_model,
                "n_heads": self.cfg.n_heads,
                "n_layers": self.cfg.n_layers,
                "d_ff": self.cfg.d_ff,
                "block_size": self.cfg.block_size,
                "pos_encoding": self.cfg.pos_encoding,
                "dropout": self.cfg.dropout,
            },
        }


STATE = ModelState()

app = FastAPI(title="llm-from-scratch UI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Request schemas
# ----------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = "ROMEO:"
    max_tokens: int = Field(300, ge=1, le=2000)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(40, ge=0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    seed: Optional[int] = None


class TokenizeRequest(BaseModel):
    text: str = ""


class AttentionRequest(BaseModel):
    prompt: str = "To be or not"


class LoadRequest(BaseModel):
    checkpoint: str


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _norm_top_k(top_k: Optional[int]) -> Optional[int]:
    return top_k if top_k and top_k > 0 else None


def _encode_prompt(prompt: str) -> torch.Tensor:
    tok = STATE.tokenizer
    # Char tokenizer raises on unknown chars; skip them so the UI never 500s.
    ids = [tok.stoi[c] for c in prompt if c in tok.stoi]
    if not ids:
        ids = [0]
    return torch.tensor(ids, dtype=torch.long, device=STATE.device).unsqueeze(0)


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "trained": STATE.checkpoint is not None}


@app.get("/api/model")
def model_info() -> dict:
    return STATE.info()


@app.get("/api/checkpoints")
def list_checkpoints() -> dict:
    items = []
    for p in sorted(CHECKPOINT_DIR.glob("*.pt")):
        items.append({"name": p.name, "size": p.stat().st_size})
    return {"checkpoints": items, "active": STATE.checkpoint}


@app.post("/api/load")
def load_checkpoint_route(req: LoadRequest) -> dict:
    try:
        STATE.load(req.checkpoint)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))
    return STATE.info()


@app.post("/api/tokenize")
def tokenize(req: TokenizeRequest) -> dict:
    tok = STATE.tokenizer
    tokens = []
    for ch in req.text:
        tokens.append({"char": ch, "id": tok.stoi.get(ch, -1)})
    return {"tokens": tokens, "count": len(tokens), "vocab_size": tok.vocab_size}


@app.post("/api/attention")
@torch.no_grad()
def attention(req: AttentionRequest) -> dict:
    model, cfg = STATE.model, STATE.cfg
    ids = _encode_prompt(req.prompt)[:, : cfg.block_size]
    logits, all_attn = model(ids)
    tokens = [STATE.tokenizer.itos[i] for i in ids[0].tolist()]

    layers = []
    for w in all_attn:               # each [B, H, T, T]
        layers.append(w[0].detach().cpu().tolist())  # [H, T, T]

    return {
        "tokens": tokens,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "attention": layers,         # [layer][head][query][key]
    }


def _generation_stream(req: GenerateRequest):
    """Yield Server-Sent-Events: one `token` event per generated character,
    then a final `done` event. Uses the simple sliding-window decode so the
    prompt + output may exceed block_size."""
    if req.seed is not None:
        torch.manual_seed(req.seed)

    model, cfg = STATE.model, STATE.cfg
    top_k = _norm_top_k(req.top_k)
    ids = _encode_prompt(req.prompt)

    meta = {"trained": STATE.checkpoint is not None, "prompt": req.prompt}
    yield f"event: start\ndata: {json.dumps(meta)}\n\n"

    for i in range(req.max_tokens):
        context = ids[:, -cfg.block_size:]
        logits, _ = model(context)
        next_logits = logits[0, -1, :]
        probs = F.softmax(next_logits / max(req.temperature, 1e-6), dim=-1)
        next_token = sample_token(next_logits, req.temperature, top_k, req.top_p)
        ids = torch.cat([ids, next_token.view(1, 1)], dim=1)

        tid = int(next_token.item())
        payload = {
            "i": i,
            "char": STATE.tokenizer.itos[tid],
            "id": tid,
            "prob": float(probs[tid].item()),
        }
        yield f"event: token\ndata: {json.dumps(payload)}\n\n"

    full = STATE.tokenizer.decode(ids[0].tolist())
    yield f"event: done\ndata: {json.dumps({'text': full})}\n\n"


@app.post("/api/generate/stream")
def generate_stream(req: GenerateRequest):
    return StreamingResponse(
        _generation_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/generate")
@torch.no_grad()
def generate(req: GenerateRequest) -> dict:
    if req.seed is not None:
        torch.manual_seed(req.seed)
    model, cfg = STATE.model, STATE.cfg
    top_k = _norm_top_k(req.top_k)
    ids = _encode_prompt(req.prompt)
    for _ in range(req.max_tokens):
        context = ids[:, -cfg.block_size:]
        logits, _ = model(context)
        next_token = sample_token(logits[0, -1, :], req.temperature, top_k, req.top_p)
        ids = torch.cat([ids, next_token.view(1, 1)], dim=1)
    return {
        "text": STATE.tokenizer.decode(ids[0].tolist()),
        "trained": STATE.checkpoint is not None,
    }
