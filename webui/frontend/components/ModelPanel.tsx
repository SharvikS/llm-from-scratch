"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Cpu, HardDrive, Layers, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type { CheckpointList, ModelInfo } from "@/lib/types";

function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

export function ModelPanel({
  info,
  onReload,
}: {
  info: ModelInfo | null;
  onReload: () => void;
}) {
  const [ckpts, setCkpts] = useState<CheckpointList | null>(null);
  const [loadingName, setLoadingName] = useState<string | null>(null);

  useEffect(() => {
    api.checkpoints().then(setCkpts).catch(() => setCkpts(null));
  }, [info]);

  const load = async (name: string) => {
    setLoadingName(name);
    try {
      await api.load(name);
      onReload();
    } finally {
      setLoadingName(null);
    }
  };

  if (!info) {
    return <div className="glass p-8 text-sm text-slate-500">Loading model…</div>;
  }

  const c = info.config;
  const stats = [
    { label: "Parameters", value: formatParams(info.num_params), icon: Cpu },
    { label: "Layers", value: c.n_layers, icon: Layers },
    { label: "Attention heads", value: c.n_heads, icon: Layers },
    { label: "Model dim", value: c.d_model, icon: Layers },
    { label: "FFN dim", value: c.d_ff, icon: Layers },
    { label: "Context", value: `${c.block_size} tok`, icon: Layers },
    { label: "Vocab", value: info.vocab_size, icon: Layers },
    { label: "Pos. encoding", value: c.pos_encoding, icon: Layers },
  ];

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
      {/* Stats + architecture diagram */}
      <div className="flex flex-col gap-6">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {stats.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
              className="glass-soft p-4"
            >
              <p className="label">{s.label}</p>
              <p className="mt-1 font-mono text-lg text-white">{s.value}</p>
            </motion.div>
          ))}
        </div>

        <div className="glass p-6">
          <p className="label mb-4">Forward pass</p>
          <div className="flex flex-col items-center gap-1.5">
            <Stage label="Token embedding" sub={`${info.vocab_size} → ${c.d_model}`} />
            <Connector />
            <Stage
              label={
                c.pos_encoding === "rope"
                  ? "RoPE (inside attention)"
                  : "Sinusoidal positional encoding"
              }
            />
            <Connector />
            {Array.from({ length: c.n_layers }).map((_, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.1 + i * 0.05 }}
                className="w-full max-w-md rounded-xl border border-accent/20 bg-accent/5 px-4 py-2.5"
              >
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-accent-soft">Block {i + 1}</span>
                  <span className="font-mono text-slate-500">
                    {c.n_heads}-head attn + FFN
                  </span>
                </div>
              </motion.div>
            ))}
            <Connector />
            <Stage label="Final LayerNorm" />
            <Connector />
            <Stage label="LM head (tied)" sub={`${c.d_model} → ${info.vocab_size}`} accent />
          </div>
        </div>
      </div>

      {/* Runtime + checkpoints */}
      <div className="flex flex-col gap-4">
        <div className="glass space-y-4 p-5">
          <div className="flex items-center justify-between">
            <p className="label">Runtime</p>
            <span
              className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
                info.trained
                  ? "bg-emerald-500/15 text-emerald-300"
                  : "bg-amber-500/15 text-amber-300"
              }`}
            >
              {info.trained ? "trained" : "demo (untrained)"}
            </span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <HardDrive className="h-4 w-4 text-slate-500" />
            <span className="text-slate-400">device</span>
            <span className="ml-auto font-mono text-accent-soft">{info.device}</span>
          </div>
          <div className="flex items-center gap-2 text-sm">
            <Cpu className="h-4 w-4 text-slate-500" />
            <span className="text-slate-400">checkpoint</span>
            <span className="ml-auto truncate font-mono text-accent-soft">
              {info.checkpoint ?? "—"}
            </span>
          </div>
        </div>

        <div className="glass space-y-3 p-5">
          <div className="flex items-center justify-between">
            <p className="label">Checkpoints</p>
            <button
              onClick={() => api.checkpoints().then(setCkpts)}
              className="text-slate-500 transition-colors hover:text-accent-soft"
              aria-label="Refresh checkpoints"
            >
              <RefreshCw className="h-3.5 w-3.5" />
            </button>
          </div>
          {!ckpts?.checkpoints.length ? (
            <p className="text-xs leading-relaxed text-slate-500">
              No <code className="font-mono">.pt</code> files in{" "}
              <code className="font-mono">checkpoints/</code> yet. Run a training
              job, then refresh.
            </p>
          ) : (
            <div className="space-y-1.5">
              {ckpts.checkpoints.map((ck) => (
                <button
                  key={ck.name}
                  onClick={() => load(ck.name)}
                  disabled={loadingName !== null}
                  className={`flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left text-sm transition-all ${
                    ckpts.active === ck.name
                      ? "border-accent/50 bg-accent/10 text-white"
                      : "border-white/5 bg-white/[0.02] text-slate-300 hover:border-white/15"
                  }`}
                >
                  <span className="truncate font-mono text-xs">{ck.name}</span>
                  <span className="ml-2 shrink-0 text-[11px] text-slate-500">
                    {loadingName === ck.name
                      ? "loading…"
                      : `${(ck.size / 1e6).toFixed(1)} MB`}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Stage({
  label,
  sub,
  accent,
}: {
  label: string;
  sub?: string;
  accent?: boolean;
}) {
  return (
    <div
      className={`w-full max-w-md rounded-xl border px-4 py-2.5 text-center ${
        accent
          ? "border-cyan-glow/30 bg-cyan-glow/5"
          : "border-white/5 bg-white/[0.02]"
      }`}
    >
      <p className="text-sm text-slate-200">{label}</p>
      {sub && <p className="font-mono text-[11px] text-slate-500">{sub}</p>}
    </div>
  );
}

function Connector() {
  return <div className="h-3 w-px bg-gradient-to-b from-white/20 to-white/5" />;
}
