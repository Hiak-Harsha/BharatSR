"use client";

import React, { useState } from "react";
import { useSuperresolve } from "../hooks/useSuperresolve";
import { usePixelProfile } from "../hooks/usePixelProfile";
import { useConsoleStore } from "@/lib/store";
import { ImageComparisonSlider } from "@/components/ui/ImageComparisonSlider";
import { PixelProfileChart } from "@/components/charts/PixelProfileChart";
import { UncertaintyScatterChart } from "@/components/charts/UncertaintyScatterChart";
import { GeospatialViewer } from "@/components/map/GeospatialViewer";
import { Play, Sparkles, Activity, ShieldCheck, Download, AlertCircle } from "lucide-react";
import { formatTime, cn } from "@/lib/utils";

export function InferencePanel({ className }: { className?: string }) {
  const selectedModel = useConsoleStore((s) => s.selectedModel);
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const customFile = useConsoleStore((s) => s.customFile);
  const currentQuality = useConsoleStore((s) => s.currentQuality);
  const currentRunId = useConsoleStore((s) => s.currentRunId);

  const [inspectedPoint, setInspectedPoint] = useState<{ x: number; y: number } | null>(null);

  const {
    mutate: runInference,
    data: srResult,
    isPending: isInferring,
    error: inferenceError,
  } = useSuperresolve();

  const {
    mutate: inspectPixel,
    data: pixelProfile,
    isPending: isProfiling,
  } = usePixelProfile();

  const handleRun = () => {
    runInference({
      sampleId: selectedSample || undefined,
      file: customFile || undefined,
      modelId: selectedModel,
      quality: currentQuality,
    });
  };

  const handlePixelClick = (x: number, y: number) => {
    setInspectedPoint({ x, y });
    inspectPixel({
      x,
      y,
      sampleId: selectedSample || undefined,
      runId: currentRunId || srResult?.run_id || undefined,
      modelId: selectedModel,
    });
  };

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      {/* Action Header Banner */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-950">
        <div>
          <h2 className="font-mono text-base font-bold text-zinc-100 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-amber-400" />
            Super-Resolution Inference Pipeline
          </h2>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            {customFile
              ? `Input: Custom upload "${customFile.name}"`
              : `Input: ${selectedSample || "No sample selected"}`}{" "}
            • Model: {selectedModel.toUpperCase()} • Mode:{" "}
            {currentQuality === "high" ? "D4 Self-Ensemble TTA (High)" : "Single Pass (Fast)"}
          </p>
        </div>

        <button
          type="button"
          disabled={isInferring || (!selectedSample && !customFile)}
          onClick={handleRun}
          className={cn(
            "inline-flex items-center gap-2 px-6 py-2.5 rounded-lg font-mono text-xs font-bold tracking-wider uppercase transition shadow-lg",
            isInferring
              ? "bg-amber-500/50 text-zinc-950 cursor-not-allowed"
              : "bg-amber-500 hover:bg-amber-400 text-zinc-950 shadow-[0_0_20px_rgba(245,158,11,0.3)] active:scale-95"
          )}
        >
          {isInferring ? (
            <>
              <Activity className="w-4 h-4 animate-spin" />
              Executing SR Pipeline...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              Execute 4× Super-Resolution
            </>
          )}
        </button>
      </div>

      {inferenceError && (
        <div className="p-4 rounded-xl border border-rose-800/60 bg-rose-950/20 text-rose-300 font-mono text-xs flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 mt-0.5" />
          <div>
            <span className="font-bold text-rose-200">Execution Error:</span>
            <p className="mt-1">{inferenceError.message}</p>
          </div>
        </div>
      )}

      {/* Primary Visual Comparison Canvas */}
      {isInferring ? (
        <div className="aspect-square w-full max-h-[580px] rounded-xl border border-zinc-800 bg-zinc-950 flex flex-col items-center justify-center p-8 gap-4">
          <div className="w-12 h-12 rounded-full border-2 border-amber-500/20 border-t-amber-400 animate-spin" />
          <div className="font-mono text-xs text-zinc-400 text-center space-y-1">
            <p className="text-zinc-200 font-bold">Executing Deep Neural Inference...</p>
            <p className="text-[11px] text-zinc-500">
              Applying Hann-windowed tile blending & heteroscedastic uncertainty estimation
            </p>
          </div>
        </div>
      ) : srResult ? (
        (() => {
          const input = srResult.input as any;
          const output = srResult.output as any;
          const bicubic = srResult.bicubic as any;
          const groundTruth = srResult.ground_truth as any;
          const errorMap = srResult.error_map as any;
          const metrics = srResult.metrics as any;
          const uncertainty = srResult.uncertainty as any;
          const geoMetadata = srResult.geospatial_metadata as any;

          return (
            <>
              <ImageComparisonSlider
                beforeSrc={input.image}
                beforeViews={input.views}
                bicubicSrc={bicubic?.image}
                bicubicViews={bicubic?.views}
                afterSrc={output.image}
                afterViews={output.views}
                groundTruthSrc={groundTruth?.image}
                groundTruthViews={groundTruth?.views}
                uncertaintySrc={uncertainty?.image}
                errorMapSrc={errorMap?.image}
                beforeLabel="Low-Resolution Input (10m)"
                afterLabel={`BharatSR (${srResult.model_id.toUpperCase()} 2.5m)`}
                inspectedPoint={inspectedPoint}
                onInspectPixel={handlePixelClick}
              />

              {/* Scientific Telemetry Strip */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
                  <span className="text-zinc-500 block text-[11px] uppercase">Latency</span>
                  <span className="text-zinc-200 font-bold text-sm">
                    {formatTime(srResult.inference_time_s)}
                  </span>
                  <span className="text-[10px] text-zinc-500 block mt-0.5">End-to-End</span>
                </div>

                <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
                  <span className="text-zinc-500 block text-[11px] uppercase">PSNR</span>
                  <span className="text-emerald-400 font-bold text-sm">
                    {metrics?.psnr?.value != null ? `${Number(metrics.psnr.value).toFixed(2)} dB` : "N/A"}
                  </span>
                  <span className="text-[10px] text-zinc-500 block mt-0.5">
                    {metrics?.psnr?.quality || "Peak Signal-to-Noise"}
                  </span>
                </div>

                <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
                  <span className="text-zinc-500 block text-[11px] uppercase">SSIM</span>
                  <span className="text-emerald-400 font-bold text-sm">
                    {metrics?.ssim?.value != null ? Number(metrics.ssim.value).toFixed(4) : "N/A"}
                  </span>
                  <span className="text-[10px] text-zinc-500 block mt-0.5">Structural Fidelity</span>
                </div>

                <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
                  <span className="text-zinc-500 block text-[11px] uppercase">SAM Angle</span>
                  <span className="text-cyan-400 font-bold text-sm">
                    {metrics?.sam?.value != null ? `${Number(metrics.sam.value).toFixed(2)}°` : "N/A"}
                  </span>
                  <span className="text-[10px] text-zinc-500 block mt-0.5">Spectral Angle Mapper</span>
                </div>

                <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
                  <span className="text-zinc-500 block text-[11px] uppercase">Downsample Cons.</span>
                  <span className="text-amber-400 font-bold text-sm">
                    {metrics?.downsample_consistency?.value != null
                      ? `${(Number(metrics.downsample_consistency.value) * 100).toFixed(1)}%`
                      : "N/A"}
                  </span>
                  <span className="text-[10px] text-zinc-500 block mt-0.5">Physics Preservation</span>
                </div>
              </div>

              {/* Interactive Inspection Charts & Geospatial View */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <PixelProfileChart
                  data={pixelProfile ?? null}
                  isLoading={isProfiling}
                />

                <UncertaintyScatterChart
                  scatter={uncertainty?.scatter}
                  summary={uncertainty?.summary}
                />
              </div>

              {/* Authentic Geospatial Viewer */}
              <GeospatialViewer
                geoMetadata={geoMetadata}
                imageUrl={output.image}
                label="Super-Resolved Output Tile"
              />
            </>
          );
        })()
      ) : (
        <div className="p-12 rounded-xl border border-dashed border-zinc-800 bg-zinc-950/40 text-center font-mono text-xs text-zinc-500 flex flex-col items-center gap-3">
          <Activity className="w-8 h-8 text-zinc-700" />
          <span>Select a Sentinel-2 sample tile or upload custom imagery, then click Execute.</span>
        </div>
      )}
    </div>
  );
}
