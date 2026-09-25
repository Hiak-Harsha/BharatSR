"use client";

import React, { useState, useEffect } from "react";
import { useSuperresolve } from "../hooks/useSuperresolve";
import { usePixelProfile } from "../hooks/usePixelProfile";
import { useJobPolling } from "@/features/jobs/hooks/useJobPolling";
import { submitAsyncSuperresolve, SuperResolveResponse } from "@/lib/api-client";
import { useConsoleStore } from "@/lib/store";
import { ImageComparisonSlider } from "@/components/ui/ImageComparisonSlider";
import { PixelProfileChart } from "@/components/charts/PixelProfileChart";
import { UncertaintyScatterChart } from "@/components/charts/UncertaintyScatterChart";
import { GeospatialViewer } from "@/components/map/GeospatialViewer";
import { Play, Sparkles, Activity, ShieldCheck, Download, AlertCircle, Clock, CheckCircle2, Crosshair } from "lucide-react";
import { formatTime, cn } from "@/lib/utils";
import { PixelResolveCanvas } from "@/components/effects/PixelResolveCanvas";
import { getSamplePreview } from "@/lib/api-client";

export function InferencePanel({ className }: { className?: string }) {
  const selectedModel = useConsoleStore((s) => s.selectedModel);
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const customFile = useConsoleStore((s) => s.customFile);
  const currentQuality = useConsoleStore((s) => s.currentQuality);
  const currentRunId = useConsoleStore((s) => s.currentRunId);
  const setCurrentRunId = useConsoleStore((s) => s.setCurrentRunId);

  const [inspectedPoint, setInspectedPoint] = useState<{ x: number; y: number } | null>(null);
  const [isAsyncMode, setIsAsyncMode] = useState<boolean>(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [asyncSubmitting, setAsyncSubmitting] = useState<boolean>(false);
  const [asyncError, setAsyncError] = useState<string | null>(null);
  const [asyncResult, setAsyncResult] = useState<SuperResolveResponse | null>(null);
  const [samplePreviewUrl, setSamplePreviewUrl] = useState<string>("/satellite_demo.png");

  useEffect(() => {
    async function loadSamplePreview() {
      const sampleId = selectedSample || "sample_real_s2";
      try {
        const p = await getSamplePreview(sampleId, 32);
        if (p?.views) {
          const url = (p.views as any).composite || p.views.sr || p.views.lr;
          if (url) setSamplePreviewUrl(url);
        }
      } catch {
        // fallback
      }
    }
    loadSamplePreview();
  }, [selectedSample]);

  const {
    mutate: runInference,
    data: syncResult,
    isPending: isSyncInferring,
    error: syncError,
  } = useSuperresolve();

  const { data: jobData } = useJobPolling(activeJobId);

  // When async job completes, parse result
  useEffect(() => {
    if (jobData?.status === "completed" && jobData.result) {
      try {
        const parsed = (typeof jobData.result === "string"
          ? JSON.parse(jobData.result)
          : jobData.result) as SuperResolveResponse;
        setAsyncResult(parsed);
        if (parsed.run_id) {
          setCurrentRunId(parsed.run_id);
        }
      } catch (e) {
        console.error("Failed to parse async job result:", e);
      }
    }
  }, [jobData, setCurrentRunId]);

  const {
    mutate: inspectPixel,
    data: pixelProfile,
    isPending: isProfiling,
  } = usePixelProfile();

  const srResult = isAsyncMode ? asyncResult : syncResult;
  const isInferring = isAsyncMode
    ? asyncSubmitting || (jobData != null && jobData.status !== "completed" && jobData.status !== "failed" && !jobData.is_cancelled)
    : isSyncInferring;
  const inferenceError = isAsyncMode ? (asyncError ? new Error(asyncError) : jobData?.error_message ? new Error(jobData.error_message) : null) : syncError;

  const handleRun = async () => {
    setAsyncError(null);
    if (!isAsyncMode) {
      runInference({
        sampleId: selectedSample || undefined,
        file: customFile || undefined,
        modelId: selectedModel,
        quality: currentQuality,
      });
    } else {
      setAsyncSubmitting(true);
      setAsyncResult(null);
      try {
        const res = await submitAsyncSuperresolve({
          sampleId: selectedSample || undefined,
          file: customFile || undefined,
          modelId: selectedModel,
          quality: currentQuality,
        });
        setActiveJobId(res.job_id);
      } catch (err: any) {
        setAsyncError(err?.message || "Failed to submit async job");
      } finally {
        setAsyncSubmitting(false);
      }
    }
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

        <div className="flex items-center gap-3">
          {/* Sync / Async Mode Toggle */}
          <div className="flex items-center gap-1.5 p-1 rounded-lg bg-zinc-900 border border-zinc-800 font-mono text-xs">
            <button
              type="button"
              onClick={() => setIsAsyncMode(false)}
              className={cn(
                "px-2.5 py-1 rounded text-[11px] font-semibold transition",
                !isAsyncMode ? "bg-zinc-800 text-amber-400 shadow" : "text-zinc-400 hover:text-zinc-200"
              )}
            >
              Direct Sync
            </button>
            <button
              type="button"
              onClick={() => setIsAsyncMode(true)}
              className={cn(
                "px-2.5 py-1 rounded text-[11px] font-semibold transition flex items-center gap-1",
                isAsyncMode ? "bg-zinc-800 text-amber-400 shadow" : "text-zinc-400 hover:text-zinc-200"
              )}
            >
              <Clock className="w-3 h-3" />
              Async Queue
            </button>
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
                {isAsyncMode ? "Processing Job..." : "Executing SR Pipeline..."}
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Execute 4× Super-Resolution
              </>
            )}
          </button>
        </div>
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
          <div className="font-mono text-xs text-zinc-400 text-center space-y-2">
            <p className="text-zinc-200 font-bold">
              {isAsyncMode
                ? `Background Worker Task: Job ${activeJobId?.slice(0, 8) || "..."}`
                : "Executing Deep Neural Inference..."}
            </p>
            {isAsyncMode && jobData && (
              <div className="w-64 max-w-full mx-auto space-y-1">
                <div className="w-full bg-zinc-900 border border-zinc-800 h-2 rounded-full overflow-hidden">
                  <div
                    className="bg-amber-400 h-full transition-all duration-300"
                    style={{ width: `${jobData.progress_pct || 15}%` }}
                  />
                </div>
                <div className="flex justify-between text-[10px] text-zinc-500">
                  <span className="capitalize">Status: {jobData.status}</span>
                  <span>{jobData.progress_pct || 0}%</span>
                </div>
              </div>
            )}
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
          const dimensions = output?.shape
            ? { width: output.shape[2], height: output.shape[1] }
            : input?.shape
            ? { width: input.shape[2] * 4, height: input.shape[1] * 4 }
            : undefined;

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
                imageDimensions={dimensions}
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
                  <span className="text-[10px] text-zinc-500 block mt-0.5">Spectral Mapper</span>
                </div>

                <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
                  <span className="text-zinc-500 block text-[11px] uppercase">Downsample MAE</span>
                  <span className="text-amber-400 font-bold text-sm">
                    {metrics?.downsample_consistency?.value != null
                      ? Number(metrics.downsample_consistency.value).toFixed(4)
                      : "N/A"}
                  </span>
                  <span className="text-[10px] text-zinc-500 block mt-0.5">Physics Preservation</span>
                </div>
              </div>

              {/* Pointwise Pixel Profile Inspector */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <PixelProfileChart data={pixelProfile ?? null} isLoading={isProfiling} />
                <UncertaintyScatterChart
                  scatter={uncertainty?.scatter}
                  summary={uncertainty?.summary}
                />
              </div>

              {/* Interactive Geospatial Viewer (if georeferenced) */}
              {geoMetadata?.has_geo && (
                <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
                  <h3 className="font-mono text-sm font-bold text-zinc-100 flex items-center gap-2 mb-3">
                    <ShieldCheck className="w-4 h-4 text-emerald-400" />
                    Interactive Geospatial GIS Footprint (EPSG:32643 / WGS 84)
                  </h3>
                  <GeospatialViewer
                    geoMetadata={geoMetadata}
                    imageUrl={output?.image || ""}
                    className="h-80"
                  />
                </div>
              )}
            </>
          );
        })()
      ) : (
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 rounded-lg bg-zinc-950 border border-zinc-800 font-mono text-xs shadow-md">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-ping" />
              <span className="text-zinc-200 font-bold">Interactive Optical Sensor Lens</span>
              <span className="text-zinc-500 hidden sm:inline">· 10m Raw S2 Blocks &rarr; 2.5m Analytical Focus</span>
            </div>
            <button
              type="button"
              onClick={handleRun}
              className="px-3.5 py-1.5 rounded-md bg-amber-500 hover:bg-amber-400 text-zinc-950 font-bold transition flex items-center gap-1.5 shadow"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Execute 4x Super-Resolution</span>
            </button>
          </div>

          <div className="relative aspect-square w-full max-h-[540px] rounded-xl overflow-hidden border border-zinc-800 bg-zinc-950 shadow-2xl">
            <PixelResolveCanvas
              src={samplePreviewUrl || "/satellite_demo.png"}
              alt="Interactive Optical Resolving Lens"
              className="w-full h-full"
              overlayLabel="Hover / drag cursor over 10m raw sensor pixels to resolve 2.5m detail"
            />
          </div>

          <div className="flex items-center justify-between text-[11px] font-mono text-zinc-500 px-1">
            <span>Tile: {selectedSample || "sample_real_s2"} (10m Native Sentinel-2 BOA Reflectance)</span>
            <span className="text-amber-400 flex items-center gap-1">
              <Crosshair className="w-3.5 h-3.5" />
              <span>Move pointer across the image to resolve pixels</span>
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
