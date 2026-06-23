"use client";

import { useCallback, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Loader2, Send, Sparkles, Square, Trash2 } from "lucide-react";
import { streamGenerate } from "@/lib/api";
import type { GenerateParams, StreamToken } from "@/lib/types";
import { Slider } from "./ui/Slider";

interface Char {
  ch: string;
  prob: number;
}

const PRESETS = [
  "ROMEO:",
  "To be, or not to be,",
  "JULIET:\nO Romeo,",
  "First Citizen:\n",
];

// Map a sampled probability to a colour: low prob -> cyan, high -> accent.
function probColor(p: number): string {
  const hue = 190 - Math.min(p, 1) * 130; // 190 (cyan) -> 60 (warm)
  return `hsl(${hue}, 85%, 72%)`;
}

export function Playground({ trained }: { trained: boolean }) {
  const [params, setParams] = useState<GenerateParams>({
    prompt: "ROMEO:",
    max_tokens: 300,
    temperature: 0.8,
    top_k: 40,
    top_p: null,
    seed: null,
  });
  const [chars, setChars] = useState<Char[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<null | (() => void)>(null);
  const outputRef = useRef<HTMLDivElement>(null);

  const set = <K extends keyof GenerateParams>(k: K, v: GenerateParams[K]) =>
    setParams((p) => ({ ...p, [k]: v }));

  const run = useCallback(() => {
    setChars([]);
    setError(null);
    setRunning(true);
    abortRef.current = streamGenerate(params, {
      onToken: (t: StreamToken) => {
        setChars((prev) => [...prev, { ch: t.char, prob: t.prob }]);
        requestAnimationFrame(() => {
          outputRef.current?.scrollTo({
            top: outputRef.current.scrollHeight,
            behavior: "smooth",
          });
        });
      },
      onDone: () => setRunning(false),
      onError: (e) => {
        setError(e.message);
        setRunning(false);
      },
    });
  }, [params]);

  const stop = () => {
    abortRef.current?.();
    setRunning(false);
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
      {/* ---- Output column ---- */}
      <div className="flex flex-col gap-4">
        <div className="glass flex flex-col">
          <div className="flex items-center gap-3 border-b border-white/5 px-5 py-3.5">
            <Sparkles className="h-4 w-4 text-accent-soft" />
            <span className="text-sm font-medium">Generation</span>
            <div className="ml-auto flex items-center gap-2 text-xs text-slate-500">
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  running ? "animate-pulse bg-cyan-glow" : "bg-slate-600"
                }`}
              />
              {chars.length} chars
            </div>
          </div>

          <div
            ref={outputRef}
            className="relative max-h-[52vh] min-h-[320px] overflow-y-auto px-6 py-5 font-mono text-[15px] leading-relaxed"
          >
            <span className="text-slate-500">{params.prompt}</span>
            {chars.map((c, i) => (
              <motion.span
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.18 }}
                style={{ color: probColor(c.prob) }}
                title={`p=${c.prob.toFixed(3)}`}
              >
                {c.ch}
              </motion.span>
            ))}
            {running && <span className="stream-caret" />}

            {!chars.length && !running && (
              <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 text-center text-slate-600">
                <Sparkles className="h-7 w-7 animate-float text-accent/40" />
                <p className="text-sm">Press generate to stream characters live.</p>
                <p className="text-xs text-slate-700">
                  Colour encodes the model&apos;s confidence in each character.
                </p>
              </div>
            )}
          </div>

          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden border-t border-red-500/20 bg-red-500/5 px-5 py-2 text-xs text-red-300"
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {!trained && (
          <div className="glass-soft flex items-start gap-3 px-4 py-3 text-xs text-amber-200/80">
            <span className="mt-0.5 text-amber-300">⚠</span>
            <p>
              No trained checkpoint found — running an <strong>untrained demo
              model</strong>, so output is random noise. Train one with{" "}
              <code className="font-mono text-amber-200">
                python scripts/train.py --config configs/tiny.yaml
              </code>{" "}
              and reload.
            </p>
          </div>
        )}
      </div>

      {/* ---- Controls column ---- */}
      <div className="flex flex-col gap-4">
        <div className="glass space-y-4 p-5">
          <div className="space-y-2">
            <label className="label">Prompt</label>
            <textarea
              value={params.prompt}
              onChange={(e) => set("prompt", e.target.value)}
              rows={3}
              className="w-full resize-none rounded-xl border border-white/10 bg-ink-800/80 px-3 py-2.5
                font-mono text-sm text-slate-200 outline-none transition-colors
                focus:border-accent/60 focus:ring-1 focus:ring-accent/40"
            />
            <div className="flex flex-wrap gap-1.5">
              {PRESETS.map((p) => (
                <button
                  key={p}
                  onClick={() => set("prompt", p)}
                  className="rounded-lg border border-white/5 bg-white/[0.03] px-2 py-1
                    text-[11px] text-slate-400 transition-colors hover:border-accent/40 hover:text-slate-200"
                >
                  {p.split("\n")[0]}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="glass space-y-5 p-5">
          <Slider
            label="Temperature"
            value={params.temperature}
            min={0}
            max={2}
            step={0.05}
            onChange={(v) => set("temperature", v)}
            format={(v) => v.toFixed(2)}
            hint="Lower = focused, higher = creative"
          />
          <Slider
            label="Top-k"
            value={params.top_k ?? 0}
            min={0}
            max={100}
            step={1}
            onChange={(v) => set("top_k", v === 0 ? null : v)}
            format={(v) => (v === 0 ? "off" : String(v))}
          />
          <Slider
            label="Top-p"
            value={params.top_p ?? 0}
            min={0}
            max={1}
            step={0.01}
            onChange={(v) => set("top_p", v === 0 ? null : v)}
            format={(v) => (v === 0 ? "off" : v.toFixed(2))}
          />
          <Slider
            label="Max tokens"
            value={params.max_tokens}
            min={16}
            max={1000}
            step={1}
            onChange={(v) => set("max_tokens", Math.round(v))}
          />
        </div>

        <div className="flex gap-2">
          {running ? (
            <button onClick={stop} className="btn-ghost flex-1">
              <Square className="h-4 w-4" /> Stop
            </button>
          ) : (
            <button onClick={run} className="btn-accent flex-1">
              {running ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
              Generate
            </button>
          )}
          <button
            onClick={() => setChars([])}
            disabled={running || !chars.length}
            className="btn-ghost px-3"
            aria-label="Clear output"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
