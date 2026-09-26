"use client";

import React from "react";
import { useCompare } from "../hooks/useCompare";
import { useConsoleStore } from "@/lib/store";
import { ImageComparisonSlider, ComparisonLayer } from "@/components/ui/ImageComparisonSlider";
import { GitCompare, Play, Activity, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export function CompareGrid({ className }: { className?: string }) {
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const customFile = useConsoleStore((s) => s.customFile);

  const {
    mutate: runCompare,
    data: compareData,
    isPending: isComparing,
    error: compareError,
  } = useCompare();

  const handleRunCompare = () => {
    runCompare({
      sampleId: selectedSample || undefined,
      file: customFile || undefined,
    });
  };

  const models = compareData?.models as Record<string, any> | undefined;
  const table = compareData?.comparison_table as any[] | undefined;
  const modelKeys = models ? Object.keys(models) : [];

  // Build dynamic layers for ImageComparisonSlider from all returned models
  const layers: ComparisonLayer[] = [];
  if (compareData?.input?.image) {
    layers.push({
      id: "input",
      label: "LR Input (10m Native)",
      tag: "10m Raw",
      image: (compareData.input as any).image as string,
      views: (compareData.input as any).views,
    });
  }
  if (models) {
    for (const [mId, mData] of Object.entries(models)) {
      const mLabel =
        mId === "bicubic"
          ? "Bicubic Interpolation"
          : mId === "srcnn"
          ? "SRCNN Baseline"
          : mId === "rcan"
          ? "BharatSR (RCAN Attention)"
          : mId === "swinir"
          ? "BharatSR (SwinIR Transformer)"
          : mId === "hat"
          ? "BharatSR (HAT Transformer)"
          : `Model ${mId.toUpperCase()}`;

      const mTag =
        mId === "bicubic"
          ? "Baseline"
          : mId === "srcnn"
          ? "CNN"
          : mId === "rcan"
          ? "Attention"
          : mId === "swinir" || mId === "hat"
          ? "Transformer"
          : "SR";

      layers.push({
        id: mId,
        label: `${mLabel} (2.5m-equivalent SR grid)`,
        tag: mTag,
        image: mData.image,
        views: mData.views,
      });
    }
  }
  if (compareData?.ground_truth?.image) {
    layers.push({
      id: "gt",
      label: "Reference Target",
      tag: "Demonstration Reference",
      image: (compareData.ground_truth as any).image as string,
      views: (compareData.ground_truth as any).views,
      isGroundTruth: true,
    });
  }

  const defaultLeft = layers.find((l) => l.id === "bicubic") ? "bicubic" : layers[0]?.id || "input";
  const defaultRight = layers.find((l) => l.id === "rcan")
    ? "rcan"
    : layers.find((l) => l.id !== "input" && l.id !== "bicubic")?.id || layers[layers.length - 1]?.id || "input";

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      {/* Action Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-950">
        <div>
          <h2 className="font-mono text-base font-bold text-zinc-100 flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-amber-400" />
            Multi-Model Architectural Benchmark
          </h2>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            Dynamically evaluate all loaded architectures (Bicubic, SRCNN, RCAN, SwinIR, HAT) on identical Sentinel-2 input
          </p>
        </div>

        <button
          type="button"
          disabled={isComparing || (!selectedSample && !customFile)}
          onClick={handleRunCompare}
          className={cn(
            "inline-flex items-center gap-2 px-6 py-2.5 rounded-lg font-mono text-xs font-bold tracking-wider uppercase transition shadow-lg",
            isComparing
              ? "bg-amber-500/50 text-zinc-950 cursor-not-allowed"
              : "bg-amber-500 hover:bg-amber-400 text-zinc-950 shadow-[0_0_20px_rgba(245,158,11,0.3)] active:scale-95"
          )}
        >
          {isComparing ? (
            <>
              <Activity className="w-4 h-4 animate-spin" />
              Benchmarking All Loaded Models...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              Run Full Comparison
            </>
          )}
        </button>
      </div>

      {compareError && (
        <div className="p-4 rounded-xl border border-rose-800/60 bg-rose-950/20 text-rose-300 font-mono text-xs flex items-start justify-between gap-3">
          <div className="flex items-start gap-3 min-w-0 flex-1">
            <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 mt-0.5" />
            <div className="min-w-0">
              <span className="font-bold text-rose-200">Comparison Notice:</span>
              <p className="mt-1 break-words">{compareError.message}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={handleRunCompare}
            disabled={isComparing}
            className="px-3 py-1.5 rounded bg-rose-900/60 hover:bg-rose-800 text-rose-100 text-xs font-bold shrink-0 transition"
          >
            Retry
          </button>
        </div>
      )}

      {isComparing ? (
        <div className="aspect-square w-full max-h-[480px] rounded-xl border border-zinc-800 bg-zinc-950 flex flex-col items-center justify-center p-8 gap-4">
          <div className="w-12 h-12 rounded-full border-2 border-amber-500/20 border-t-amber-400 animate-spin" />
          <div className="font-mono text-xs text-zinc-300 text-center space-y-1">
            <p className="text-zinc-100 font-bold text-sm">Evaluating Model Architectures...</p>
            <p className="text-xs text-zinc-400">
              Generating bicubic baseline, running SRCNN, RCAN, and loaded Transformer architectures
            </p>
          </div>
        </div>
      ) : compareData ? (
        <div className="flex flex-col gap-6">
          {/* Plain-English Takeaway Lead Banner */}
          {(() => {
            const rankings = (compareData.rankings as any[]) || [];
            if (rankings.length === 0) return null;
            const top = rankings[0];
            return (
              <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-950/20 font-mono text-xs text-amber-300">
                <span className="font-bold text-amber-200 uppercase tracking-wider block mb-1">
                  Architectural Benchmark Takeaway:
                </span>
                <p className="text-zinc-200 leading-relaxed font-sans text-xs">
                  <strong className="text-amber-300 uppercase">{top?.model_id}</strong> leads the benchmark on this scene with composite score of <strong className="text-amber-300">{(Number(top?.composite_score || 0) * 100).toFixed(1)}/100</strong>, delivering superior structural retention and radiometric fidelity over baseline bicubic interpolation.
                </p>
              </div>
            );
          })()}

          {/* Dynamic Multi-Layer Comparison Slider */}
          <ImageComparisonSlider
            layers={layers}
            initialLeftId={defaultLeft}
            initialRightId={defaultRight}
          />

          {/* Dynamic Quantitative Comparison Matrix */}
          {table && table.length > 0 && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-950 overflow-hidden p-4 shadow-xl">
              <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-zinc-200">
                  Quantitative Comparison Matrix (Dynamically Populated)
                </h3>
                <span className="font-mono text-[10px] text-zinc-500">
                  PSNR / SSIM / UIQI: Higher is better &bull; SAM / ERGAS: Lower is better
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs">
                  <thead>
                    <tr className="border-b border-zinc-800 text-zinc-400 text-xs">
                      <th className="py-3 px-3">Metric</th>
                      <th className="py-3 px-3">Unit</th>
                      {modelKeys.map((mKey) => (
                        <th key={mKey} className="py-3 px-3 uppercase text-zinc-200">
                          {mKey}
                        </th>
                      ))}
                      <th className="py-3 px-3 text-amber-300">Dynamic Leader</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-850">
                    {table.map((row: any, idx: number) => {
                      return (
                        <tr key={idx} className="hover:bg-zinc-900/40">
                          <td className="py-3 px-3 font-medium text-zinc-200">{row.metric}</td>
                          <td className="py-3 px-3 text-zinc-400">{row.unit}</td>
                          {modelKeys.map((mKey) => {
                            const val = row[mKey];
                            const isBest = row.best_model === mKey;
                            return (
                              <td
                                key={mKey}
                                className={cn(
                                  "py-3 px-3",
                                  isBest ? "text-amber-300 font-bold" : "text-zinc-300"
                                )}
                              >
                                {val != null ? (typeof val === "number" ? val.toFixed(3) : val) : "—"}
                              </td>
                            );
                          })}
                          <td className="py-3 px-3">
                            {row.best_model && (
                              <span className="px-2 py-0.5 rounded text-xs uppercase font-bold bg-zinc-800 text-amber-300 border border-zinc-700">
                                {row.best_model}
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="p-12 rounded-xl border border-dashed border-zinc-800 bg-zinc-950/40 text-center font-mono text-xs text-zinc-400 flex flex-col items-center gap-3">
          <GitCompare className="w-8 h-8 text-zinc-600" />
          <p className="text-zinc-200 font-medium text-sm max-w-lg">
            Directly test how physics-constrained neural models compare against standard bicubic interpolation.
          </p>
          <span className="text-zinc-400 text-xs max-w-md">
            Runs Bicubic, SRCNN, RCAN, and loaded Transformer architectures on identical Sentinel-2 input to measure edge sharpness, spectral angle distortion, and sensor downsample fidelity.
          </span>
          <button
            type="button"
            onClick={handleRunCompare}
            disabled={!selectedSample && !customFile}
            className="mt-2 px-6 py-2.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-zinc-950 font-bold transition shadow-lg active:scale-95"
          >
            Run Full Multi-Model Comparison
          </button>
        </div>
      )}
    </div>
  );
}
