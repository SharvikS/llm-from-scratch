# Web UI — `llm-from-scratch`

A modern, animated playground for the hand-built Transformer. It is a thin
layer on top of the existing model code — nothing about the architecture,
tokenizer, or sampling is re-implemented here.

```
webui/
├── backend/          FastAPI server that loads a checkpoint and serves the model
│   ├── server.py
│   └── requirements.txt
└── frontend/         Next.js 14 (App Router) + Tailwind + Framer Motion
    ├── app/
    ├── components/
    └── lib/
```

## What you get

| Tab          | What it does |
|--------------|--------------|
| **Playground** | Streams generated characters live over SSE. Sliders for temperature / top-k / top-p / max-tokens. Each character is tinted by the model's confidence. |
| **Attention**  | Computes real `[layer][head][query][key]` attention for a prompt and renders an interactive, hoverable heatmap. Switch layer/head instantly. |
| **Tokenizer**  | Live character-level tokenization with per-id colour coding. |
| **Model**      | Parameter count, full config, an animated forward-pass diagram, device/runtime info, and a checkpoint switcher. |

If no trained checkpoint exists under `checkpoints/`, the backend falls back to
a freshly-initialised **demo model** so the whole UI still works (generation is
random noise until you train and reload).

## Run it

Two processes — backend on `:8000`, frontend on `:3000`.

### 1. Backend

From the **repo root** (so `model`, `tokenizer`, … import correctly):

```bash
pip install -r requirements.txt                 # torch, numpy, … (once)
pip install -r webui/backend/requirements.txt   # fastapi, uvicorn
uvicorn webui.backend.server:app --reload --port 8000
```

### 2. Frontend

```bash
cd webui/frontend
npm install
npm run dev
```

Open <http://localhost:3000>. The dev server proxies `/api/*` to the backend
(see `next.config.mjs`), so there are no CORS issues and SSE streams pass
straight through. To point at a non-default backend:

```bash
cp .env.example .env.local   # then edit API_BASE_URL
```

## How it maps onto the codebase

| UI feature        | Backend route            | Underlying code |
|-------------------|--------------------------|-----------------|
| Streaming gen     | `POST /api/generate/stream` | `model(context)` + `inference.sampler.sample_token`, sliding-window decode |
| Attention heatmap | `POST /api/attention`    | `Transformer.forward` → `all_attn_weights` |
| Tokenizer         | `POST /api/tokenize`     | `tokenizer.char_tokenizer.CharTokenizer` |
| Model card        | `GET /api/model`         | `ModelConfig` + `Transformer.num_params()` |
| Checkpoint switch | `POST /api/load`         | `training.checkpointing.load_checkpoint` |

## Production build

```bash
# backend
uvicorn webui.backend.server:app --host 0.0.0.0 --port 8000
# frontend
cd webui/frontend && npm run build && npm run start
```
