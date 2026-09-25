"use client";

import React, { useState } from "react";
import { useDownstreamMasks } from "../hooks/useDownstreamMasks";
import { useConsoleStore } from "@/lib/store";
import { Target, Play, Activity, AlertCircle, Layers, CheckCircle2, TrendingUp, Info, ShieldAlert } from "lucide-react";
import { cn } from "@/lib/utils";

export function DownstreamMasksPanel({ className }: { className?: string }) {
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const selectedModel = useConsoleStore((s) => s.selectedModel);
  const currentRunId = useConsoleStore((s) => s.currentRunId);

  const [activeTaskKey, setActiveTaskKey] = useState<string>("micro_canopy");
  const [activeMaskType, setActiveMaskType] = useState<string>("sr_mask");

  const {
    data: downstreamData,
    isLoading,
    refetch,
    error,
  } = useDownstreamMasks(
    currentRunId ? undefined : (selectedSample || undefined),
    currentRunId || undefined,
    selectedModel,
    Boolean(currentRunId || selectedSample)
  );

  const tasks = downstreamData?.tasks || {};
  const currentTask = tasks[activeTaskKey];

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-950">
        <div>
          <h2 className="font-mono text-base font-bold text-zinc-100 flex items-center gap-2">
            <Target className="w-5 h-5 text-indigo-400" />
            Downstream Consistency Analysis
          </h2>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            {currentRunId
              ? `Pseudo-label consistency evaluation for run ${currentRunId}`
              : "Pseudo-label consistency evaluation comparing model outputs against rule-derived reference masks"}
          </p>
        </div>

        <button
          type="button"
          disabled={isLoading || (!currentRunId && !selectedSample)}
          onClick={() => refetch()}
          className={cn(
            "inline-flex items-center gap-2 px-6 py-2.5 rounded-lg font-mono text-xs font-bold tracking-wider uppercase transition shadow-lg",
            isLoading
              ? "bg-indigo-500/50 text-zinc-950 cursor-not-allowed"
              : "bg-indigo-500 hover:bg-indigo-400 text-zinc-950 shadow-[0_0_20px_rgba(99,102,241,0.3)] active:scale-95"
          )}
        >
          {isLoading ? (
            <>
              <Activity className="w-4 h-4 animate-spin" />
              Evaluating Consistency...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              Compute Downstream Consistency
            </>
          )}
        </button>
      </div>

      {/* Prominent Mandatory Methodology Disclaimer */}
      <div className="rounded-lg bg-amber-950/30 border border-amber-800/50 p-3.5 text-xs font-mono text-amber-300 flex items-start gap-2.5">
        <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-amber-200">Pseudo-Label Methodology Notice: </span>
          <span>
            These metrics compare model outputs against rule-derived reference masks and are not independent labelled benchmark results.
          </span>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl border border-rose-800/60 bg-rose-950/20 text-rose-300 font-mono text-xs flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 mt-0.5" />
          <div>
            <span className="font-bold text-rose-200">Evaluation Failed:</span>
            <p className="mt-1">{error.message}</p>
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="aspect-video w-full rounded-xl border border-zinc-800 bg-zinc-950 flex flex-col items-center justify-center p-8 gap-4">
          <div className="w-12 h-12 rounded-full border-2 border-indigo-500/20 border-t-indigo-400 animate-spin" />
          <div className="font-mono text-xs text-zinc-300 text-center">
            Running rule-based canopy extraction and evaluating pseudo-label consistency...
          </div>
        </div>
      ) : currentTask ? (
        <div className="flex flex-col gap-6">
          {/* Task Selector Tabs */}
          <div className="flex items-center gap-2 border-b border-zinc-800 pb-3">
            {Object.entries(tasks).map(([k, t]) => (
              <button
                key={k}
                type="button"
                onClick={() => {
                  setActiveTaskKey(k);
                  setActiveMaskType("sr_mask");
                }}
                className={cn(
                  "px-4 py-2 rounded-lg font-mono text-xs transition flex items-center gap-2",
                  activeTaskKey === k
                    ? "bg-indigo-950/80 text-indigo-300 border border-indigo-700/60 font-bold"
                    : "bg-zinc-900 text-zinc-400 hover:text-zinc-200 border border-zinc-800"
                )}
              >
                <Layers className="w-4 h-4" />
                {t.task_name}
              </button>
            ))}
          </div>

          {/* Metric Uplift Spotlight */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {(() => {
              const rcanPrec = (currentTask.rcan?.precision ?? 0) * 100;
              const bicPrec = (currentTask.bicubic?.precision ?? 0) * 100;
              const delta = rcanPrec - bicPrec;
              return (
                <div className="p-3.5 rounded-xl border border-zinc-800 bg-zinc-950 font-mono">
                  <span className="text-zinc-400 text-xs block uppercase font-semibold">Precision Uplift</span>
                  <div className="flex items-baseline gap-2 mt-1">
                    <span className="text-emerald-400 font-bold text-lg">
                      {rcanPrec.toFixed(1)}%
                    </span>
                    <span className="text-zinc-500 text-xs line-through">
                      {bicPrec.toFixed(1)}%
                    </span>
                  </div>
                  <span className="text-xs text-emerald-400/90 block mt-1 flex items-center gap-1 font-semibold">
                    <TrendingUp className="w-3.5 h-3.5" />
                    {delta >= 0 ? `+${delta.toFixed(2)}%` : `${delta.toFixed(2)}%`} vs Bicubic
                  </span>
                </div>
              );
            })()}

            <div className="p-3.5 rounded-xl border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-400 text-xs block uppercase font-semibold">IoU (Intersection/Union)</span>
              <div className="flex items-baseline gap-2 mt-1">
                <span className="text-indigo-400 font-bold text-lg">
                  {((currentTask.rcan?.iou ?? 0) * 100).toFixed(1)}%
                </span>
                <span className="text-zinc-500 text-xs line-through">
                  {((currentTask.bicubic?.iou ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              <span className="text-xs text-zinc-400 block mt-1">Spatial overlap score</span>
            </div>

            <div className="p-3.5 rounded-xl border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-400 text-xs block uppercase font-semibold">F1-Score</span>
              <div className="flex items-baseline gap-2 mt-1">
                <span className="text-cyan-400 font-bold text-lg">
                  {((currentTask.rcan?.f1 ?? 0) * 100).toFixed(1)}%
                </span>
                <span className="text-zinc-500 text-xs line-through">
                  {((currentTask.bicubic?.f1 ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              <span className="text-xs text-zinc-400 block mt-1">Harmonic mean</span>
            </div>

            <div className="p-3.5 rounded-xl border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-400 text-xs block uppercase font-semibold">Target Pixels</span>
              <div className="flex items-center gap-2 mt-1 text-emerald-400 font-bold text-base">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>{currentTask.ground_truth_pixel_count.toLocaleString()}</span>
              </div>
              <span className="text-xs text-zinc-400 block mt-1">
                Reference mask pixels
              </span>
            </div>
          </div>

          {/* Mask Visualizer Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-3 flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-semibold text-emerald-400">
                  BharatSR 4× SR Mask
                </span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-800/50">
                  Precision: {((currentTask.rcan?.precision ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="aspect-square w-full rounded bg-black overflow-hidden border border-zinc-800">
                {currentTask.masks?.sr_mask ? (
                  <img
                    src={currentTask.masks.sr_mask}
                    alt="BharatSR SR Mask"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-xs text-zinc-600 font-mono">
                    Mask not available
                  </div>
                )}
              </div>
            </div>

            <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-3 flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-semibold text-zinc-400">
                  Bicubic 4× Baseline Mask
                </span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-zinc-900 text-zinc-400 border border-zinc-800">
                  Precision: {((currentTask.bicubic?.precision ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="aspect-square w-full rounded bg-black overflow-hidden border border-zinc-800">
                {currentTask.masks?.bicubic_mask ? (
                  <img
                    src={currentTask.masks.bicubic_mask}
                    alt="Bicubic Mask"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-xs text-zinc-600 font-mono">
                    Mask not available
                  </div>
                )}
              </div>
            </div>

            <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-3 flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-semibold text-indigo-400">
                  Rule-Derived Reference Mask
                </span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-indigo-950/60 text-indigo-300 border border-indigo-800/50">
                  Target: 100%
                </span>
              </div>
              <div className="aspect-square w-full rounded bg-black overflow-hidden border border-zinc-800">
                {currentTask.masks?.ground_truth_mask ? (
                  <img
                    src={currentTask.masks.ground_truth_mask}
                    alt="Reference Mask"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-xs text-zinc-600 font-mono">
                    Reference not available
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Description & Scientific Methodology */}
          <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950/70 font-mono text-xs text-zinc-300 flex items-start gap-3">
            <Info className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
            <div>
              <span className="font-bold text-zinc-100 block mb-1">
                {currentTask.task_name} — Methodology
              </span>
              <p className="text-zinc-400 leading-relaxed">{currentTask.description}</p>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-12 rounded-xl border border-dashed border-zinc-800 bg-zinc-950/40 text-center font-mono text-xs text-zinc-400 flex flex-col items-center gap-3">
          <Target className="w-8 h-8 text-zinc-600" />
          <span className="text-zinc-300 font-medium">
            No downstream consistency analysis has been generated for the current run.
          </span>
          <button
            type="button"
            onClick={() => refetch()}
            disabled={!currentRunId && !selectedSample}
            className="mt-2 px-4 py-2 rounded bg-indigo-500 hover:bg-indigo-400 text-zinc-950 font-bold transition"
          >
            Compute Downstream Consistency
          </button>
        </div>
      )}
    </div>
  );
}
