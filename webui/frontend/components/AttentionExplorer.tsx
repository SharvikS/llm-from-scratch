"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Eye, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import type { AttentionResult } from "@/lib/types";

function cellColor(v: number): string {
  // 0 -> transparent ink, 1 -> bright accent
  return `rgba(124, 92, 255, ${Math.max(0.04, v)})`;
}

function display(tok: string): string {
  if (tok === " ") return "␣";
  if (tok === "\n") return "↵";
  if (tok === "\t") return "⇥";
  return tok;
}

export function AttentionExplorer() {
  const [prompt, setPrompt] = useState("To be or not");
  const [data, setData] = useState<AttentionResult | null>(null);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [hover, setHover] = useState<{ q: number; k: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAttention = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.attention(prompt);
      setData(res);
      setLayer(0);
      setHead(0);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAttention();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const matrix = useMemo(() => {
    if (!data) return null;
    return data.attention[layer]?.[head] ?? null;
  }, [data, layer, head]);

  const tokens = data?.tokens ?? [];
  const n = tokens.length;

  return (
    <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
      {/* Controls */}
      <div className="flex flex-col gap-4">
        <div className="glass space-y-3 p-5">
          <label className="label">Prompt to inspect</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            className="w-full resize-none rounded-xl border border-white/10 bg-ink-800/80 px-3 py-2.5
              font-mono text-sm outline-none focus:border-accent/60 focus:ring-1 focus:ring-accent/40"
          />
          <button onClick={fetchAttention} disabled={loading} className="btn-accent w-full">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Eye className="h-4 w-4" />}
            Compute attention
          </button>
          {error && <p className="text-xs text-red-300">{error}</p>}
        </div>

        {data && (
          <div className="glass space-y-4 p-5">
            <div>
              <p className="label mb-2">Layer · {layer + 1}/{data.n_layers}</p>
              <div className="flex flex-wrap gap-1.5">
                {Array.from({ length: data.n_layers }).map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setLayer(i)}
                    className={`h-8 w-8 rounded-lg text-xs font-medium transition-all ${
                      layer === i
                        ? "bg-accent text-white shadow-glow"
                        : "bg-white/[0.03] text-slate-400 hover:bg-white/[0.07]"
                    }`}
                  >
                    {i + 1}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <p className="label mb-2">Head · {head + 1}/{data.n_heads}</p>
              <div className="flex flex-wrap gap-1.5">
                {Array.from({ length: data.n_heads }).map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setHead(i)}
                    className={`h-8 w-8 rounded-lg text-xs font-medium transition-all ${
                      head === i
                        ? "bg-cyan-glow text-ink-900 shadow-glow"
                        : "bg-white/[0.03] text-slate-400 hover:bg-white/[0.07]"
                    }`}
                  >
                    {i + 1}
                  </button>
                ))}
              </div>
            </div>
            <p className="text-xs leading-relaxed text-slate-500">
              Each row is a query token; columns are the keys it attends to.
              Causal masking keeps the upper triangle dark — tokens never look
              ahead.
            </p>
          </div>
        )}
      </div>

      {/* Heatmap */}
      <div className="glass flex min-h-[400px] items-center justify-center p-6">
        {!matrix ? (
          <div className="text-sm text-slate-600">No attention computed yet.</div>
        ) : (
          <motion.div
            key={`${layer}-${head}`}
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.25 }}
            className="w-full overflow-auto"
          >
            <div className="inline-grid" style={{ gridTemplateColumns: `auto repeat(${n}, 1fr)` }}>
              {/* top-left corner */}
              <div />
              {/* column headers */}
              {tokens.map((t, j) => (
                <div
                  key={`c${j}`}
                  className={`px-1 pb-1 text-center font-mono text-[10px] transition-colors ${
                    hover?.k === j ? "text-cyan-glow" : "text-slate-500"
                  }`}
                >
                  {display(t)}
                </div>
              ))}

              {matrix.map((row, q) => (
                <FragmentRow
                  key={`r${q}`}
                  q={q}
                  row={row}
                  token={tokens[q]}
                  hover={hover}
                  setHover={setHover}
                />
              ))}
            </div>

            <div className="mt-5 flex items-center gap-3 text-xs text-slate-500">
              <span>0.0</span>
              <div className="h-2 w-40 rounded-full bg-gradient-to-r from-ink-600 to-accent" />
              <span>1.0</span>
              {hover && (
                <span className="ml-3 font-mono text-accent-soft">
                  “{display(tokens[hover.q])}” → “{display(tokens[hover.k])}” ={" "}
                  {matrix[hover.q][hover.k].toFixed(3)}
                </span>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}

function FragmentRow({
  q,
  row,
  token,
  hover,
  setHover,
}: {
  q: number;
  row: number[];
  token: string;
  hover: { q: number; k: number } | null;
  setHover: (h: { q: number; k: number } | null) => void;
}) {
  return (
    <>
      <div
        className={`pr-2 text-right font-mono text-[10px] leading-none transition-colors ${
          hover?.q === q ? "text-accent-soft" : "text-slate-500"
        } flex items-center justify-end`}
      >
        {display(token)}
      </div>
      {row.map((v, k) => (
        <div
          key={k}
          onMouseEnter={() => setHover({ q, k })}
          onMouseLeave={() => setHover(null)}
          className="aspect-square min-h-[14px] min-w-[14px] rounded-[3px] transition-transform hover:scale-110 hover:ring-1 hover:ring-white/40"
          style={{ backgroundColor: cellColor(v) }}
          title={`${v.toFixed(3)}`}
        />
      ))}
    </>
  );
}
