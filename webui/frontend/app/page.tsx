"use client";

import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Boxes, Grid3x3, type LucideIcon, Sparkles, Type } from "lucide-react";
import { api } from "@/lib/api";
import type { ModelInfo } from "@/lib/types";
import { Playground } from "@/components/Playground";
import { AttentionExplorer } from "@/components/AttentionExplorer";
import { TokenizerLab } from "@/components/TokenizerLab";
import { ModelPanel } from "@/components/ModelPanel";

type TabId = "playground" | "attention" | "tokenizer" | "model";

const TABS: { id: TabId; label: string; icon: LucideIcon }[] = [
  { id: "playground", label: "Playground", icon: Sparkles },
  { id: "attention", label: "Attention", icon: Grid3x3 },
  { id: "tokenizer", label: "Tokenizer", icon: Type },
  { id: "model", label: "Model", icon: Boxes },
];

export default function Home() {
  const [tab, setTab] = useState<TabId>("playground");
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);

  const loadInfo = useCallback(async () => {
    try {
      const data = await api.model();
      setInfo(data);
      setOnline(true);
    } catch {
      setOnline(false);
    }
  }, []);

  useEffect(() => {
    loadInfo();
  }, [loadInfo]);

  return (
    <main className="mx-auto flex min-h-screen max-w-6xl flex-col px-5 pb-16 pt-8">
      <Header info={info} online={online} />

      {/* Tab bar */}
      <nav className="mt-8 flex w-full gap-1 rounded-2xl border border-white/5 bg-ink-800/50 p-1.5 backdrop-blur-xl sm:w-fit">
        {TABS.map((t) => {
          const Icon = t.icon;
          const active = tab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`relative flex flex-1 items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-colors sm:flex-none ${
                active ? "text-white" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {active && (
                <motion.span
                  layoutId="tab-pill"
                  className="absolute inset-0 rounded-xl bg-accent/20 shadow-glow ring-1 ring-accent/40"
                  transition={{ type: "spring", stiffness: 400, damping: 32 }}
                />
              )}
              <Icon className="relative z-10 h-4 w-4" />
              <span className="relative z-10 hidden sm:inline">{t.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Panels */}
      <div className="mt-6 flex-1">
        {online === false && (
          <div className="glass p-8 text-center">
            <p className="text-sm text-red-300">Backend unreachable.</p>
            <p className="mt-2 text-xs text-slate-500">
              Start it with{" "}
              <code className="font-mono text-slate-400">
                uvicorn webui.backend.server:app --port 8000
              </code>{" "}
              from the repo root.
            </p>
          </div>
        )}

        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.22 }}
          >
            {tab === "playground" && (
              <Playground trained={info?.trained ?? false} />
            )}
            {tab === "attention" && <AttentionExplorer />}
            {tab === "tokenizer" && <TokenizerLab />}
            {tab === "model" && <ModelPanel info={info} onReload={loadInfo} />}
          </motion.div>
        </AnimatePresence>
      </div>
    </main>
  );
}

function Header({
  info,
  online,
}: {
  info: ModelInfo | null;
  online: boolean | null;
}) {
  return (
    <header className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-3">
        <div className="relative flex h-11 w-11 items-center justify-center rounded-2xl bg-accent/15 ring-1 ring-accent/40">
          <span className="absolute inset-0 animate-pulse-ring rounded-2xl ring-1 ring-accent/40" />
          <Boxes className="h-5 w-5 text-accent-soft" />
        </div>
        <div>
          <h1 className="text-lg font-semibold tracking-tight">
            <span className="gradient-text">llm-from-scratch</span>
          </h1>
          <p className="text-xs text-slate-500">
            A Transformer built by hand in pure PyTorch
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 text-xs">
        <span
          className={`h-2 w-2 rounded-full ${
            online == null
              ? "bg-slate-600"
              : online
                ? "animate-pulse bg-emerald-400"
                : "bg-red-400"
          }`}
        />
        <span className="text-slate-400">
          {online == null
            ? "connecting…"
            : online
              ? `${info?.device ?? ""} · ${info?.trained ? "trained" : "demo"}`
              : "offline"}
        </span>
      </div>
    </header>
  );
}
