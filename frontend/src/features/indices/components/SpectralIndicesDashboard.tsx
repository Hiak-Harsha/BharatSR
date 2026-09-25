"use client";

import React, { useState } from "react";
import { useIndices } from "../hooks/useIndices";
import { useConsoleStore } from "@/lib/store";
import { BarChart3, Play, Activity, AlertCircle, Eye, EyeOff } from "lucide-react";
import { cn } from "@/lib/utils";

const INDEX_DESCRIPTIONS: Record<string, { fullName: string; formula: string; purpose: string }> = {
  ndvi: {
    fullName: "Normalized Difference Vegetation Index",
    formula: "(B8 - B4) / (B8 + B4)",
    purpose: "Quantifies photosynthetic activity, crop canopy vigor, and vegetative biomass density.",
  },
  ndwi: {
    fullName: "Normalized Difference Water Index",
    formula: "(B3 - B8) / (B3 + B8)",
    purpose: "Delineates open surface water bodies, wetlands, and canopy water stress levels.",
  },
  ndre: {
    fullName: "Normalized Difference Red Edge",
    formula: "(B8 - B5) / (B8 + B5)",
    purpose: "Measures chlorophyll concentration and nitrogen status in dense agricultural canopies.",
  },
  evi: {
    fullName: "Enhanced Vegetation Index",
    formula: "2.5 × (B8 - B4) / (B8 + 6×B4 - 7.5×B2 + 1)",
    purpose: "Reduces atmospheric aerosol scattering and soil background noise in high-biomass regions.",
  },
  savi: {
    fullName: "Soil-Adjusted Vegetation Index",
    formula: "1.5 × (B8 - B4) / (B8 + B4 + 0.5)",
    purpose: "Corrects for soil spectral reflectance in arid, semi-arid, or early-stage planting zones.",
  },
};

export function SpectralIndicesDashboard({ className }: { className?: string }) {
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const selectedModel = useConsoleStore((s) => s.selectedModel);
  const currentRunId = useConsoleStore((s) => s.currentRunId);

  const [activeHeatmapIndex, setActiveHeatmapIndex] = useState<string | null>(null);

  const {
    mutate: computeIndices,
    data: indicesData,
    isPending: isComputing,
    error: indicesError,
  } = useIndices();

  const handleCompute = () => {
    computeIndices({
      runId: currentRunId || undefined,
      sampleId: currentRunId ? undefined : (selectedSample || undefined),
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
            Spectral Indices Preservation Analysis
          </h2>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            {currentRunId
              ? `Evaluating completed run ${currentRunId} (${selectedModel.toUpperCase()})`
              : selectedSample
              ? `Evaluating scene ${selectedSample} (${selectedModel.toUpperCase()})`
              : "Select a scene or complete super-resolution run"}
          </p>
        </div>

        <button
          type="button"
          disabled={isComputing || (!currentRunId && !selectedSample)}
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
              Computing Indices...
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
          <div className="font-mono text-xs text-zinc-300 text-center">
            Extracting NIR (B8), Red (B4), Green (B3), Blue (B2) and evaluating non-linear spectral ratios...
          </div>
        </div>
      ) : indicesData && Object.keys(indices).length > 0 ? (
        <div className="flex flex-col gap-6">
          {/* Clean Summary Table */}
          <div className="rounded-xl border border-zinc-800 bg-zinc-950 overflow-hidden shadow-xl">
            <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
              <h3 className="font-mono text-sm font-bold text-zinc-200 uppercase tracking-wider">
                Spectral Indices Summary Table
              </h3>
              <span className="font-mono text-xs text-zinc-500">
                5 Standard Bio-Physical Indicators
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead className="bg-zinc-900/60 border-b border-zinc-800 text-zinc-400 uppercase text-[11px]">
                  <tr>
                    <th className="py-3 px-4">Index</th>
                    <th className="py-3 px-4">LR Mean</th>
                    <th className="py-3 px-4">BharatSR Mean</th>
                    <th className="py-3 px-4">Difference (Δ)</th>
                    <th className="py-3 px-4 text-right">Heatmap Inspection</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-850">
                  {Object.entries(indices).map(([name, stat]) => {
                    const lrMean = stat.lr?.mean ?? 0;
                    const srMean = stat.sr?.mean ?? 0;
                    const diff = srMean - lrMean;
                    const isInspecting = activeHeatmapIndex === name;

                    return (
                      <tr key={name} className="hover:bg-zinc-900/40 transition">
                        <td className="py-3 px-4">
                          <div className="font-bold text-amber-300 uppercase text-sm">
                            {name}
                          </div>
                          <div className="text-[11px] font-sans font-semibold text-zinc-200 mt-0.5">
                            {INDEX_DESCRIPTIONS[name.toLowerCase()]?.fullName || name}
                          </div>
                          <div className="text-[10px] font-sans text-zinc-400 mt-0.5 max-w-md leading-relaxed">
                            {INDEX_DESCRIPTIONS[name.toLowerCase()]?.purpose || "Spectral ratio indicator"}
                          </div>
                        </td>
                        <td className="py-3 px-4 text-zinc-300">
                          {lrMean.toFixed(4)}
                        </td>
                        <td className="py-3 px-4 text-emerald-400 font-medium">
                          {srMean.toFixed(4)}
                        </td>
                        <td className="py-3 px-4 text-zinc-400">
                          <span className={cn(
                            "px-1.5 py-0.5 rounded text-[11px]",
                            Math.abs(diff) < 0.02
                              ? "bg-emerald-950/60 text-emerald-300"
                              : "bg-amber-950/60 text-amber-300"
                          )}>
                            {diff >= 0 ? `+${diff.toFixed(4)}` : diff.toFixed(4)}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-right">
                          <button
                            type="button"
                            onClick={() => setActiveHeatmapIndex(isInspecting ? null : name)}
                            className={cn(
                              "inline-flex items-center gap-1.5 px-3 py-1 rounded border text-xs transition",
                              isInspecting
                                ? "bg-amber-500 text-zinc-950 border-amber-400 font-bold"
                                : "bg-zinc-900 text-zinc-300 border-zinc-700 hover:border-zinc-500"
                            )}
                          >
                            {isInspecting ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                            <span>{isInspecting ? "Close Heatmap" : "View Heatmap"}</span>
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Progressive Disclosure Heatmap Viewer */}
          {activeHeatmapIndex && indices[activeHeatmapIndex] && (
            <div className="p-4 rounded-xl border border-amber-500/40 bg-zinc-950 flex flex-col gap-4 animate-in fade-in duration-200">
              <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
                <div className="flex items-center gap-2">
                  <Eye className="w-4 h-4 text-amber-400" />
                  <span className="font-mono text-sm font-bold text-amber-300 uppercase">
                    {activeHeatmapIndex} Spatial Distribution Heatmap
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setActiveHeatmapIndex(null)}
                  className="font-mono text-xs text-zinc-400 hover:text-zinc-200"
                >
                  ✕ Close
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex flex-col gap-2">
                  <span className="font-mono text-xs text-zinc-400">
                    Low-Resolution 10m Input ({activeHeatmapIndex})
                  </span>
                  <div className="aspect-square bg-black rounded-lg overflow-hidden border border-zinc-800 relative">
                    {indices[activeHeatmapIndex].lr_visualization ? (
                      <img
                        src={indices[activeHeatmapIndex].lr_visualization}
                        alt={`LR ${activeHeatmapIndex}`}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-zinc-600 font-mono text-xs">
                        No visualization available
                      </div>
                    )}
                  </div>
                  <span className="font-mono text-xs text-zinc-500">
                    LR Mean: {indices[activeHeatmapIndex].lr?.mean?.toFixed(4)} &bull; Std: {indices[activeHeatmapIndex].lr?.std?.toFixed(4) ?? "N/A"}
                  </span>
                </div>

                <div className="flex flex-col gap-2">
                  <span className="font-mono text-xs text-amber-300 font-semibold">
                    BharatSR 2.5m-equivalent Output ({activeHeatmapIndex})
                  </span>
                  <div className="aspect-square bg-black rounded-lg overflow-hidden border border-amber-500/40 relative">
                    {indices[activeHeatmapIndex].sr_visualization ? (
                      <img
                        src={indices[activeHeatmapIndex].sr_visualization}
                        alt={`SR ${activeHeatmapIndex}`}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-zinc-600 font-mono text-xs">
                        No visualization available
                      </div>
                    )}
                  </div>
                  <span className="font-mono text-xs text-emerald-400 font-semibold">
                    BharatSR Mean: {indices[activeHeatmapIndex].sr?.mean?.toFixed(4)} &bull; Std: {indices[activeHeatmapIndex].sr?.std?.toFixed(4) ?? "N/A"}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="p-12 rounded-xl border border-dashed border-zinc-800 bg-zinc-950/40 text-center font-mono text-xs text-zinc-400 flex flex-col items-center gap-3">
          <BarChart3 className="w-8 h-8 text-zinc-600" />
          <span className="text-zinc-300 font-medium">
            No spectral analysis has been generated for the current run.
          </span>
          <button
            type="button"
            onClick={handleCompute}
            disabled={!currentRunId && !selectedSample}
            className="mt-2 px-4 py-2 rounded bg-amber-500 hover:bg-amber-400 text-zinc-950 font-bold transition"
          >
            Compute Spectral Indices
          </button>
        </div>
      )}
    </div>
  );
}
