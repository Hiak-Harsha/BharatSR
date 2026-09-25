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
    <div className={cn("flex flex-col gap-4 rounded-xl border border-zinc-800 bg-zinc-950 p-5", className)}>
      <div className="flex items-center justify-between border-b border-zinc-850 pb-3">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-amber-400" />
          <h3 className="font-mono text-sm font-semibold uppercase tracking-wider text-zinc-100">
            Super-Resolution Model
          </h3>
        </div>
        <span className="font-mono text-xs text-zinc-400">4× Spatial Upscaling</span>
      </div>

      {isLoading ? (
        <div className="space-y-3">
          <div className="h-16 bg-zinc-900 animate-pulse rounded-lg" />
          <div className="h-16 bg-zinc-900 animate-pulse rounded-lg" />
        </div>
      ) : isError ? (
        <div className="text-xs font-mono text-rose-400 p-3 bg-rose-950/20 border border-rose-800/40 rounded-lg">
          Failed to load models list from backend.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-2.5">
          {models?.map((model) => {
            const isSelected = selectedModel === model.id;
            const supportsUncertainty = Boolean(model.supports_uncertainty ?? model.has_uncertainty);
            return (
              <button
                key={model.id}
                type="button"
                onClick={() => setSelectedModel(model.id)}
                className={cn(
                  "flex items-start justify-between p-3.5 rounded-lg border text-left transition-all",
                  isSelected
                    ? "border-amber-500/80 bg-amber-950/20 shadow-[0_0_12px_rgba(245,158,11,0.15)] ring-1 ring-amber-500/40"
                    : "border-zinc-800 bg-zinc-900/60 hover:bg-zinc-850 hover:border-zinc-700"
                )}
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm font-bold text-zinc-100">{model.name}</span>
                    {isSelected && <CheckCircle2 className="w-4 h-4 text-amber-400" />}
                  </div>
                  <p className="font-mono text-xs text-zinc-400">
                    {model.architecture}
                  </p>
                  <div className="pt-1 flex flex-wrap items-center gap-2">
                    <span className="text-xs font-mono text-zinc-400">
                      4× SR · {model.n_bands} bands
                    </span>
                    {supportsUncertainty && (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-cyan-950/80 text-cyan-300 border border-cyan-800/60">
                        Predicted uncertainty
                      </span>
                    )}
                  </div>
                </div>

                <span
                  className={cn(
                    "text-xs font-mono uppercase px-2.5 py-1 rounded font-medium",
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

      {/* Quality Mode Toggle (Fast vs 4-way Flip TTA) */}
      <div className="mt-2 pt-3 border-t border-zinc-850 space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs font-semibold text-zinc-300">Inference Mode:</span>
          <div className="inline-flex rounded-lg border border-zinc-800 bg-zinc-900 p-1 text-xs font-mono">
            <button
              type="button"
              onClick={() => setCurrentQuality("fast")}
              className={cn(
                "px-3 py-1.5 rounded transition font-medium",
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
                "px-3 py-1.5 rounded transition font-medium",
                currentQuality === "high"
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/50"
                  : "text-zinc-400 hover:text-zinc-200"
              )}
            >
              High Quality (4-way Flip TTA)
            </button>
          </div>
        </div>
        <p className="font-mono text-xs text-zinc-500 leading-relaxed">
          {currentQuality === "high"
            ? "4-way Self-Ensemble: Runs identity, horizontal flip, vertical flip, and composite flip (approx. 4× inference work)."
            : "Standard single forward pass for minimal latency and interactive workflows."}
        </p>
      </div>
    </div>
  );
}
