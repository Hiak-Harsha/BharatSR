"use client";

import React from "react";
import { useCompare } from "../hooks/useCompare";
import { useConsoleStore } from "@/lib/store";
import { ImageComparisonSlider } from "@/components/ui/ImageComparisonSlider";
import { GitCompare, Play, Activity, AlertCircle, CheckCircle2 } from "lucide-react";
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

  // Build layers for composable ImageComparisonSlider
  const layers = compareData
    ? [
        ...(compareData.input?.image
          ? [
              {
                id: "input",
                label: "LR Input (10m)",
                tag: "Raw",
                image: (compareData.input as any).image as string,
                views: (compareData.input as any).views,
              },
            ]
          : []),
        ...(models?.bicubic
          ? [
              {
                id: "bicubic",
                label: "Bicubic Interpolation",
                tag: "Baseline",
                image: models.bicubic.image,
                views: models.bicubic.views,
              },
            ]
          : []),
        ...(models?.srcnn
          ? [
              {
                id: "srcnn",
                label: "SRCNN Baseline",
                tag: "CNN",
                image: models.srcnn.image,
                views: models.srcnn.views,
              },
            ]
          : []),
        ...(models?.rcan
          ? [
              {
                id: "rcan",
                label: "BharatSR (RCAN Attention)",
                tag: "Deep SR",
                image: models.rcan.image,
                views: models.rcan.views,
              },
            ]
          : []),
        ...(compareData.ground_truth?.image
          ? [
              {
                id: "gt",
                label: "Ground Truth Reference",
                tag: "GT",
                image: (compareData.ground_truth as any).image as string,
                views: (compareData.ground_truth as any).views,
                isGroundTruth: true,
              },
            ]
          : []),
      ]
    : [];

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
            Compare Bicubic vs SRCNN vs BharatSR (RCAN) side-by-side on identical Sentinel-2 input
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
              Benchmarking Architectures...
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
        <div className="p-4 rounded-xl border border-rose-800/60 bg-rose-950/20 text-rose-300 font-mono text-xs flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 mt-0.5" />
          <div>
            <span className="font-bold text-rose-200">Comparison Failed:</span>
            <p className="mt-1">{compareError.message}</p>
          </div>
        </div>
      )}

      {isComparing ? (
        <div className="aspect-square w-full max-h-[480px] rounded-xl border border-zinc-800 bg-zinc-950 flex flex-col items-center justify-center p-8 gap-4">
          <div className="w-12 h-12 rounded-full border-2 border-amber-500/20 border-t-amber-400 animate-spin" />
          <div className="font-mono text-xs text-zinc-400 text-center space-y-1">
            <p className="text-zinc-200 font-bold">Evaluating Model Ensembles...</p>
            <p className="text-[11px] text-zinc-500">
              Generating bicubic baseline, running SRCNN forward pass, and computing RCAN channel attention
            </p>
          </div>
        </div>
      ) : compareData ? (
        <>
          {/* Multi-Layer Comparison Slider */}
          <ImageComparisonSlider
            layers={layers}
            initialLeftId="bicubic"
            initialRightId="rcan"
          />

          {/* Analytical Benchmark Comparison Table */}
          {table && table.length > 0 && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-950 overflow-hidden p-4">
              <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-zinc-200 mb-3">
                Quantitative Comparison Matrix
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs">
                  <thead>
                    <tr className="border-b border-zinc-800 text-zinc-500 text-[11px]">
                      <th className="py-2.5 px-3">Metric</th>
                      <th className="py-2.5 px-3">Unit</th>
                      <th className="py-2.5 px-3">Bicubic</th>
                      <th className="py-2.5 px-3">SRCNN</th>
                      <th className="py-2.5 px-3 text-amber-300">BharatSR (RCAN)</th>
                      <th className="py-2.5 px-3">Best Model</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-zinc-850">
                    {table.map((row: any, idx: number) => {
                      const isRcanBest = row.best_model === "rcan";
                      return (
                        <tr key={idx} className="hover:bg-zinc-900/40">
                          <td className="py-2.5 px-3 font-medium text-zinc-200">{row.metric}</td>
                          <td className="py-2.5 px-3 text-zinc-500">{row.unit}</td>
                          <td className="py-2.5 px-3 text-zinc-400">
                            {row.bicubic != null ? Number(row.bicubic).toFixed(3) : "—"}
                          </td>
                          <td className="py-2.5 px-3 text-zinc-400">
                            {row.srcnn != null ? Number(row.srcnn).toFixed(3) : "—"}
                          </td>
                          <td className="py-2.5 px-3 text-amber-300 font-bold">
                            {row.rcan != null ? Number(row.rcan).toFixed(3) : "—"}
                          </td>
                          <td className="py-2.5 px-3">
                            {row.best_model && (
                              <span
                                className={cn(
                                  "px-2 py-0.5 rounded text-[10px] uppercase font-bold",
                                  isRcanBest
                                    ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                                    : "bg-zinc-800 text-zinc-400"
                                )}
                              >
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
        </>
      ) : (
        <div className="p-12 rounded-xl border border-dashed border-zinc-800 bg-zinc-950/40 text-center font-mono text-xs text-zinc-500 flex flex-col items-center gap-3">
          <GitCompare className="w-8 h-8 text-zinc-700" />
          <span>Click "Run Full Comparison" to benchmark Bicubic vs SRCNN vs BharatSR.</span>
        </div>
      )}
    </div>
  );
}
