"use client";

import React from "react";
import { useIndices } from "../hooks/useIndices";
import { useConsoleStore } from "@/lib/store";
import { SpectralIndicesChart } from "@/components/charts/SpectralIndicesChart";
import { BarChart3, Play, Activity, AlertCircle, Eye } from "lucide-react";
import { cn } from "@/lib/utils";

export function SpectralIndicesDashboard({ className }: { className?: string }) {
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const selectedModel = useConsoleStore((s) => s.selectedModel);
  const currentRunId = useConsoleStore((s) => s.currentRunId);

  const {
    mutate: computeIndices,
    data: indicesData,
    isPending: isComputing,
    error: indicesError,
  } = useIndices();

  const handleCompute = () => {
    computeIndices({
      sampleId: selectedSample || undefined,
      runId: currentRunId || undefined,
      modelId: selectedModel,
    });
  };

  const indices = (indicesData?.indices || {}) as Record<
    string,
    {
      sr: { mean: number; std?: number; p25?: number; p75?: number };
      lr: { mean: number; std?: number };
      sr_visualization?: string;
      lr_visualization?: string;
    }
  >;

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-950">
        <div>
          <h2 className="font-mono text-base font-bold text-zinc-100 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-amber-400" />
            Bio-Physical & Spectral Indices Analysis
          </h2>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            Rigorous evaluation of NDVI, EVI, NDWI, SAVI, and NDRE preservation after 4× super-resolution
          </p>
        </div>

        <button
          type="button"
          disabled={isComputing}
          onClick={handleCompute}
          className={cn(
            "inline-flex items-center gap-2 px-6 py-2.5 rounded-lg font-mono text-xs font-bold tracking-wider uppercase transition shadow-lg",
            isComputing
              ? "bg-amber-500/50 text-zinc-950 cursor-not-allowed"
              : "bg-amber-500 hover:bg-amber-400 text-zinc-950 shadow-[0_0_20px_rgba(245,158,11,0.3)] active:scale-95"
          )}
        >
          {isComputing ? (
            <>
              <Activity className="w-4 h-4 animate-spin" />
              Computing Bio-Physical Indices...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              Compute Spectral Indices
            </>
          )}
        </button>
      </div>

      {indicesError && (
        <div className="p-4 rounded-xl border border-rose-800/60 bg-rose-950/20 text-rose-300 font-mono text-xs flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 mt-0.5" />
          <div>
            <span className="font-bold text-rose-200">Indices Calculation Error:</span>
            <p className="mt-1">{indicesError.message}</p>
          </div>
        </div>
      )}

      {isComputing ? (
        <div className="aspect-video w-full rounded-xl border border-zinc-800 bg-zinc-950 flex flex-col items-center justify-center p-8 gap-4">
          <div className="w-12 h-12 rounded-full border-2 border-amber-500/20 border-t-amber-400 animate-spin" />
          <div className="font-mono text-xs text-zinc-400 text-center">
            Extracting NIR (B8), Red (B4), Green (B3), Blue (B2) and evaluating non-linear spectral ratios...
          </div>
        </div>
      ) : indicesData ? (
        <>
          {/* Recharts Distribution Chart */}
          <SpectralIndicesChart indices={indices} />

          {/* Cards for each index with visual heatmaps */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(indices).map(([name, stat]) => (
              <div
                key={name}
                className="flex flex-col gap-3 p-4 rounded-xl border border-zinc-800 bg-zinc-950"
              >
                <div className="flex items-center justify-between border-b border-zinc-850 pb-2">
                  <span className="font-mono text-sm font-bold text-amber-300 uppercase">
                    {name}
                  </span>
                  <span className="text-[10px] font-mono text-zinc-500">
                    Δ: {((stat.sr?.mean ?? 0) - (stat.lr?.mean ?? 0)).toFixed(4)}
                  </span>
                </div>

                {/* Heatmap Visualizations */}
                <div className="grid grid-cols-2 gap-2 aspect-video bg-black rounded-lg overflow-hidden border border-zinc-850">
                  <div className="relative h-full flex flex-col justify-end p-1.5">
                    {stat.lr_visualization ? (
                      <img
                        src={stat.lr_visualization}
                        alt={`LR ${name}`}
                        className="absolute inset-0 w-full h-full object-cover"
                      />
                    ) : (
                      <div className="absolute inset-0 bg-zinc-900" />
                    )}
                    <span className="relative z-10 text-[9px] font-mono bg-black/80 px-1 py-0.5 rounded text-zinc-400">
                      LR: {stat.lr?.mean?.toFixed(3)}
                    </span>
                  </div>

                  <div className="relative h-full flex flex-col justify-end p-1.5">
                    {stat.sr_visualization ? (
                      <img
                        src={stat.sr_visualization}
                        alt={`SR ${name}`}
                        className="absolute inset-0 w-full h-full object-cover"
                      />
                    ) : (
                      <div className="absolute inset-0 bg-zinc-900" />
                    )}
                    <span className="relative z-10 text-[9px] font-mono bg-black/80 px-1 py-0.5 rounded text-amber-300">
                      BharatSR: {stat.sr?.mean?.toFixed(3)}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-zinc-400 pt-1">
                  <div>
                    <span className="text-zinc-600 block text-[9px]">LR STD</span>
                    <span>{stat.lr?.std?.toFixed(4) ?? "N/A"}</span>
                  </div>
                  <div>
                    <span className="text-zinc-600 block text-[9px]">SR STD</span>
                    <span>{stat.sr?.std?.toFixed(4) ?? "N/A"}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="p-12 rounded-xl border border-dashed border-zinc-800 bg-zinc-950/40 text-center font-mono text-xs text-zinc-500 flex flex-col items-center gap-3">
          <BarChart3 className="w-8 h-8 text-zinc-700" />
          <span>Click "Compute Spectral Indices" to inspect NDVI, EVI, and NDWI distributions.</span>
        </div>
      )}
    </div>
  );
}
