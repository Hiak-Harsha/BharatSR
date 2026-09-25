"use client";

import React from "react";
import { useCropHealth } from "../hooks/useCropHealth";
import { useConsoleStore } from "@/lib/store";
import { Sprout, Play, Activity, AlertCircle, ShieldAlert, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

export function CropHealthMap({ className }: { className?: string }) {
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const selectedModel = useConsoleStore((s) => s.selectedModel);
  const currentRunId = useConsoleStore((s) => s.currentRunId);

  const {
    mutate: runCropHealth,
    data: healthData,
    isPending: isAnalyzing,
    error: healthError,
  } = useCropHealth();

  const handleAnalyze = () => {
    runCropHealth({
      runId: currentRunId || undefined,
      sampleId: currentRunId ? undefined : (selectedSample || undefined),
      modelId: selectedModel,
    });
  };

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-950">
        <div>
          <h2 className="font-mono text-base font-bold text-zinc-100 flex items-center gap-2">
            <Sprout className="w-5 h-5 text-emerald-400" />
            Crop Health Indicators
          </h2>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            {currentRunId
              ? `Rule-based multi-spectral canopy indicators for run ${currentRunId}`
              : "Rule-based multi-spectral canopy indicators derived from 4× super-resolved reflectance"}
          </p>
        </div>

        <button
          type="button"
          disabled={isAnalyzing || (!currentRunId && !selectedSample)}
          onClick={handleAnalyze}
          className={cn(
            "inline-flex items-center gap-2 px-6 py-2.5 rounded-lg font-mono text-xs font-bold tracking-wider uppercase transition shadow-lg",
            isAnalyzing
              ? "bg-emerald-500/50 text-zinc-950 cursor-not-allowed"
              : "bg-emerald-500 hover:bg-emerald-400 text-zinc-950 shadow-[0_0_20px_rgba(16,185,129,0.3)] active:scale-95"
          )}
        >
          {isAnalyzing ? (
            <>
              <Activity className="w-4 h-4 animate-spin" />
              Evaluating Indicators...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              Compute Crop Health Indicators
            </>
          )}
        </button>
      </div>

      {/* Prominent Mandatory Scientific Disclaimer */}
      <div className="rounded-lg bg-amber-950/30 border border-amber-800/50 p-3.5 text-xs font-mono text-amber-300 flex items-start gap-2.5">
        <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-amber-200">Scientific Disclaimer: </span>
          <span>
            Rule-based spectral indicator. Not a certified agronomic assessment. Recommendations are rule-based interpretations and not cadastral or ground-verified diagnoses.
          </span>
        </div>
      </div>

      {healthError && (
        <div className="p-4 rounded-xl border border-rose-800/60 bg-rose-950/20 text-rose-300 font-mono text-xs flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 mt-0.5" />
          <div>
            <span className="font-bold text-rose-200">Analysis Error:</span>
            <p className="mt-1">{healthError.message}</p>
          </div>
        </div>
      )}

      {isAnalyzing ? (
        <div className="aspect-video w-full rounded-xl border border-zinc-800 bg-zinc-950 flex flex-col items-center justify-center p-8 gap-4">
          <div className="w-12 h-12 rounded-full border-2 border-emerald-500/20 border-t-emerald-400 animate-spin" />
          <div className="font-mono text-xs text-zinc-300 text-center">
            Performing multi-spectral thresholding across NDVI and EVI bands...
          </div>
        </div>
      ) : healthData ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Classification Map */}
          <div className="lg:col-span-2 rounded-xl border border-zinc-800 bg-zinc-950 p-4 flex flex-col gap-3">
            <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-zinc-200">
              Crop Health Zonation Map (2.5m-equivalent SR grid)
            </h3>
            <div className="aspect-square w-full rounded-lg overflow-hidden border border-zinc-850 bg-black flex items-center justify-center">
              <img
                src={healthData.classification_map}
                alt="Crop Health Map"
                className="w-full h-full object-contain"
              />
            </div>
          </div>

          {/* Area Stats & Agronomic Recommendations */}
          <div className="flex flex-col gap-4">
            <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950 font-mono text-xs space-y-3">
              <h3 className="font-semibold uppercase tracking-wider text-zinc-200 border-b border-zinc-850 pb-2">
                Spectral Indicator Metrics
              </h3>

              <div className="grid grid-cols-2 gap-2.5">
                <div className="bg-zinc-900/60 p-3 rounded border border-zinc-800">
                  <span className="text-zinc-400 text-xs block">Health Score</span>
                  <span className="text-emerald-400 font-bold text-lg">
                    {healthData.health_score.toFixed(1)}/100
                  </span>
                </div>
                <div className="bg-zinc-900/60 p-3 rounded border border-zinc-800">
                  <span className="text-zinc-400 text-xs block">NDVI Uplift</span>
                  <span className="text-cyan-400 font-bold text-lg">
                    +{(healthData.sr_vs_lr_ndvi_uplift * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="bg-zinc-900/60 p-3 rounded border border-zinc-800">
                  <span className="text-zinc-400 text-xs block">Mean NDVI</span>
                  <span className="text-zinc-100 font-bold text-sm">
                    {healthData.mean_ndvi.toFixed(3)}
                  </span>
                </div>
                <div className="bg-zinc-900/60 p-3 rounded border border-zinc-800">
                  <span className="text-zinc-400 text-xs block">Mean EVI</span>
                  <span className="text-zinc-100 font-bold text-sm">
                    {healthData.mean_evi.toFixed(3)}
                  </span>
                </div>
              </div>
            </div>

            {/* Area Distribution */}
            <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950 font-mono text-xs space-y-2.5">
              <h3 className="font-semibold uppercase tracking-wider text-zinc-200 border-b border-zinc-850 pb-2">
                Class Distribution
              </h3>
              <div className="space-y-2">
                {Object.entries(healthData.area_statistics || {}).map(([key, stat]: [string, any]) => (
                  <div key={key} className="flex items-center justify-between text-xs">
                    <span className="text-zinc-300 capitalize">{key.replace(/_/g, " ")}:</span>
                    <span className="font-bold text-zinc-100">
                      {(stat.percentage ?? 0).toFixed(1)}% ({stat.pixel_count} px)
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Recommendations */}
            {healthData.recommendations && healthData.recommendations.length > 0 && (
              <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950 font-mono text-xs space-y-2">
                <h3 className="font-semibold uppercase tracking-wider text-zinc-200 border-b border-zinc-850 pb-2">
                  Rule-based Interpretation
                </h3>
                <ul className="space-y-2 text-zinc-300 text-xs">
                  {healthData.recommendations.map((rec, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="p-12 rounded-xl border border-dashed border-zinc-800 bg-zinc-950/40 text-center font-mono text-xs text-zinc-400 flex flex-col items-center gap-3">
          <Sprout className="w-8 h-8 text-zinc-600" />
          <span className="text-zinc-300 font-medium">
            No crop health analysis has been generated for the current run.
          </span>
          <button
            type="button"
            onClick={handleAnalyze}
            disabled={!currentRunId && !selectedSample}
            className="mt-2 px-4 py-2 rounded bg-emerald-500 hover:bg-emerald-400 text-zinc-950 font-bold transition"
          >
            Compute Crop Health Indicators
          </button>
        </div>
      )}
    </div>
  );
}
