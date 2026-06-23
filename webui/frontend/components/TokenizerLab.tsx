"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api } from "@/lib/api";
import type { TokenizeResult } from "@/lib/types";

// Deterministic pastel per token id so the same char always gets one colour.
function idColor(id: number): string {
  const hue = (id * 47) % 360;
  return `hsl(${hue}, 70%, 30%)`;
}

function display(ch: string): string {
  if (ch === " ") return "·";
  if (ch === "\n") return "⏎";
  if (ch === "\t") return "⇥";
  return ch;
}

export function TokenizerLab() {
  const [text, setText] = useState("To be, or not to be.");
  const [result, setResult] = useState<TokenizeResult | null>(null);

  useEffect(() => {
    const handle = setTimeout(async () => {
      try {
        setResult(await api.tokenize(text));
      } catch {
        /* ignore */
      }
    }, 150);
    return () => clearTimeout(handle);
  }, [text]);

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="glass space-y-3 p-5">
        <label className="label">Input text</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={8}
          className="w-full resize-none rounded-xl border border-white/10 bg-ink-800/80 px-3 py-2.5
            font-mono text-sm outline-none focus:border-accent/60 focus:ring-1 focus:ring-accent/40"
        />
        <div className="flex gap-4 text-xs text-slate-500">
          <span>
            <span className="font-mono text-accent-soft">{result?.count ?? 0}</span> tokens
          </span>
          <span>
            vocab <span className="font-mono text-accent-soft">{result?.vocab_size ?? "—"}</span>
          </span>
          <span className="text-slate-600">character-level</span>
        </div>
      </div>

      <div className="glass space-y-4 p-5">
        <label className="label">Token stream</label>
        <div className="flex flex-wrap gap-1.5">
          {result?.tokens.map((t, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.15, delay: Math.min(i * 0.004, 0.4) }}
              className="group flex flex-col items-center overflow-hidden rounded-md border border-white/5"
              style={{ backgroundColor: idColor(t.id) }}
            >
              <span className="px-2 py-1 font-mono text-sm text-white">
                {display(t.char)}
              </span>
              <span className="w-full bg-black/30 px-2 py-0.5 text-center font-mono text-[10px] text-white/60">
                {t.id}
              </span>
            </motion.div>
          ))}
          {!result?.tokens.length && (
            <p className="text-sm text-slate-600">Type something to tokenize it.</p>
          )}
        </div>
      </div>
    </div>
  );
}
