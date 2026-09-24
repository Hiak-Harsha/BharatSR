"use client";

import React from "react";
import { useModels } from "../hooks/useModels";
import { useConsoleStore } from "@/lib/store";
import { Cpu, Zap, ShieldAlert, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

export function ModelSelector({ className }: { className?: string }) {
  const { data: models, isLoading, isError } = useModels();
  const selectedModel = useConsoleStore((s) => s.selectedModel);
  const setSelectedModel = useConsoleStore((s) => s.setSelectedModel);
  const currentQuality = useConsoleStore((s) => s.currentQuality);
  const setCurrentQuality = useConsoleStore((s) => s.setCurrentQuality);

  return (
    <div className={cn("flex flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-950 p-4", className)}>
      <div className="flex items-center justify-between border-b border-zinc-850 pb-2.5">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-amber-400" />
          <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-zinc-200">
            Model Architecture
          </h3>
        </div>
        <span className="font-mono text-[10px] text-zinc-500">4× Spatial Scale</span>
      </div>

      {isLoading ? (
        <div className="space-y-2">
          <div className="h-14 bg-zinc-900 animate-pulse rounded-lg" />
          <div className="h-14 bg-zinc-900 animate-pulse rounded-lg" />
        </div>
      ) : isError ? (
        <div className="text-xs font-mono text-rose-400 p-3 bg-rose-950/20 border border-rose-800/40 rounded-lg">
          Failed to load models list from backend.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-2">
          {models?.map((model) => {
            const isSelected = selectedModel === model.id;
            return (
              <button
                key={model.id}
                type="button"
                onClick={() => setSelectedModel(model.id)}
                className={cn(
                  "flex items-start justify-between p-3 rounded-lg border text-left transition-all",
                  isSelected
                    ? "border-amber-500/80 bg-amber-950/20 shadow-[0_0_12px_rgba(245,158,11,0.15)]"
                    : "border-zinc-800 bg-zinc-900/60 hover:bg-zinc-850 hover:border-zinc-700"
                )}
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs font-bold text-zinc-100">{model.name}</span>
                    {isSelected && <CheckCircle2 className="w-3.5 h-3.5 text-amber-400" />}
                  </div>
                  <p className="mt-1 font-mono text-[11px] text-zinc-400 leading-tight">
                    {model.architecture}
                  </p>
                  <div className="mt-2 flex items-center gap-2">
                    {model.has_uncertainty && (
                      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono bg-cyan-950/80 text-cyan-300 border border-cyan-800/60">
                        Heteroscedastic σ
                      </span>
                    )}
                    <span className="text-[10px] font-mono text-zinc-500">
                      {model.n_bands} Bands (B2/B3/B4/B8)
                    </span>
                  </div>
                </div>

                <span
                  className={cn(
                    "text-[10px] font-mono uppercase px-2 py-0.5 rounded",
                    model.status === "loaded"
                      ? "text-emerald-400 bg-emerald-950/60 border border-emerald-800/50"
                      : "text-zinc-500 bg-zinc-800"
                  )}
                >
                  {model.status}
                </span>
              </button>
            );
          })}
        </div>
      )}

      {/* Quality Mode Toggle (Fast vs High TTA) */}
      <div className="mt-1 pt-3 border-t border-zinc-850 flex items-center justify-between">
        <span className="font-mono text-xs text-zinc-400">Inference Mode:</span>
        <div className="inline-flex rounded-lg border border-zinc-800 bg-zinc-900 p-0.5 text-xs font-mono">
          <button
            type="button"
            onClick={() => setCurrentQuality("fast")}
            className={cn(
              "px-2.5 py-1 rounded transition",
              currentQuality === "fast"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/50"
                : "text-zinc-400 hover:text-zinc-200"
            )}
          >
            Fast (Single Pass)
          </button>
          <button
            type="button"
            onClick={() => setCurrentQuality("high")}
            className={cn(
              "px-2.5 py-1 rounded transition",
              currentQuality === "high"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/50"
                : "text-zinc-400 hover:text-zinc-200"
            )}
          >
            High (D4 Ensemble)
          </button>
        </div>
      </div>
    </div>
  );
}
