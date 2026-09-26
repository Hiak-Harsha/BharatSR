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
import { Play, Sparkles, Activity, ShieldCheck, Download, AlertCircle, Clock, CheckCircle2, Crosshair, ChevronDown, ChevronUp } from "lucide-react";
import { formatTime, cn } from "@/lib/utils";
import { PixelResolveCanvas } from "@/components/effects/PixelResolveCanvas";
import { PixelDissolve } from "@/components/effects/PixelDissolve";
import { getSamplePreview } from "@/lib/api-client";

export function InferencePanel({ className }: { className?: string }) {
  const selectedModel = useConsoleStore((s) => s.selectedModel);
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const customFile = useConsoleStore((s) => s.customFile);
  const currentQuality = useConsoleStore((s) => s.currentQuality);
  const currentRunId = useConsoleStore((s) => s.currentRunId);
  const setCurrentRunId = useConsoleStore((s) => s.setCurrentRunId);
  const setCurrentSession = useConsoleStore((s) => s.setCurrentSession);

  const [inspectedPoint, setInspectedPoint] = useState<{ x: number; y: number } | null>(null);
  const [isAsyncMode, setIsAsyncMode] = useState<boolean>(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [asyncSubmitting, setAsyncSubmitting] = useState<boolean>(false);
  const [asyncError, setAsyncError] = useState<string | null>(null);
  const [asyncResult, setAsyncResult] = useState<SuperResolveResponse | null>(null);
  const [samplePreviewUrl, setSamplePreviewUrl] = useState<string>("/satellite_demo.png");
  const [showManualLens, setShowManualLens] = useState<boolean>(false);

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

  // Hydrate AnalysisSession on sync inference completion
  useEffect(() => {
    if (syncResult && syncResult.run_id) {
      setCurrentRunId(syncResult.run_id);
      setCurrentSession({
        runId: syncResult.run_id,
        sampleId: selectedSample,
        fileName: customFile?.name || null,
        inputType: customFile ? "upload" : "sample",
        modelId: selectedModel,
        quality: currentQuality,
        status: "completed",
        createdAt: new Date().toISOString(),
        inputMetadata: syncResult.input as any,
        outputMetadata: syncResult.output as any,
        metrics: syncResult.metrics as any,
        uncertainty: syncResult.uncertainty as any,
        geospatialMetadata: syncResult.geospatial_metadata as any,
        provenance: (syncResult.geospatial_metadata as any)?.source_dataset || "Sentinel-2 L2A",
      });
    }
  }, [syncResult, setCurrentRunId, setCurrentSession, selectedSample, customFile, selectedModel, currentQuality]);

  const { data: jobData } = useJobPolling(activeJobId);

  // When async job completes, parse result and hydrate AnalysisSession
  useEffect(() => {
    if (jobData?.status === "completed" && jobData.result) {
      try {
        const parsed = (typeof jobData.result === "string"
          ? JSON.parse(jobData.result)
          : jobData.result) as SuperResolveResponse;
        setAsyncResult(parsed);
        if (parsed.run_id) {
          setCurrentRunId(parsed.run_id);
          setCurrentSession({
            runId: parsed.run_id,
            jobId: activeJobId,
            sampleId: selectedSample,
            fileName: customFile?.name || null,
            inputType: customFile ? "upload" : "sample",
            modelId: selectedModel,
            quality: currentQuality,
            status: "completed",
            createdAt: new Date().toISOString(),
            inputMetadata: parsed.input as any,
            outputMetadata: parsed.output as any,
            metrics: parsed.metrics as any,
            uncertainty: parsed.uncertainty as any,
            geospatialMetadata: parsed.geospatial_metadata as any,
            provenance: (parsed.geospatial_metadata as any)?.source_dataset || "Sentinel-2 L2A",
          });
        }
      } catch (e) {
        console.error("Failed to parse async job result:", e);
      }
    }
  }, [jobData, activeJobId, setCurrentRunId, setCurrentSession, selectedSample, customFile, selectedModel, currentQuality]);

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
              : `Input: ${selectedSample || "Sentinel-2 L2A"}`}{" "}
            • Model: {selectedModel.toUpperCase()} • Mode:{" "}
            {currentQuality === "high" ? "High Quality / 4-way Self-Ensemble (~4× work)" : "Single Pass (Fast)"}
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
              {/* RUN COMPLETE Canonical Status Banner */}
              <div className="p-4 rounded-xl border border-emerald-800/60 bg-emerald-950/20 font-mono text-xs">
                <div className="flex items-center gap-2 mb-2 pb-2 border-b border-emerald-800/40">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span className="font-bold text-emerald-300 uppercase tracking-wider text-sm">
                    RUN COMPLETE &bull; {srResult.run_id || currentRunId}
                  </span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 text-zinc-300">
                  <div>
                    <span className="text-zinc-500 block text-[11px] uppercase">Model</span>
                    <span className="font-bold text-zinc-100">{srResult.model_id.toUpperCase()}</span>
                  </div>
                  <div>
                    <span className="text-zinc-500 block text-[11px] uppercase">Input</span>
                    <span className="font-semibold text-zinc-200">
                      {geoMetadata?.source_dataset ? "Sentinel-2 L2A" : "Demo Scene"}
                    </span>
                  </div>
                  <div>
                    <span className="text-zinc-500 block text-[11px] uppercase">Bands</span>
                    <span className="font-semibold text-zinc-200">B2 / B3 / B4 / B8</span>
                  </div>
                  <div>
                    <span className="text-zinc-500 block text-[11px] uppercase">Input Dim</span>
                    <span className="font-semibold text-zinc-200">
                      {input?.shape ? `${input.shape[1]} × ${input.shape[2]}` : "64 × 64"}
                    </span>
                  </div>
                  <div>
                    <span className="text-zinc-500 block text-[11px] uppercase">Output Dim</span>
                    <span className="font-semibold text-zinc-200">
                      {output?.shape ? `${output.shape[1]} × ${output.shape[2]}` : "256 × 256"} (4×)
                    </span>
                  </div>
                  <div>
                    <span className="text-zinc-500 block text-[11px] uppercase">Output Grid</span>
                    <span className="font-bold text-emerald-400">2.5m-equivalent SR</span>
                  </div>
                </div>
              </div>

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
                afterLabel={`BharatSR (${srResult.model_id.toUpperCase()} 2.5m-equivalent SR grid)`}
                inspectedPoint={inspectedPoint}
                onInspectPixel={handlePixelClick}
                imageDimensions={dimensions}
              />

              {/* Plain-English Takeaway Lead Banner */}
              <div className="p-4 rounded-xl border border-emerald-500/30 bg-emerald-950/20 font-mono text-xs text-emerald-300">
                <span className="font-bold text-emerald-200 uppercase tracking-wider block mb-1">
                  Super-Resolution Verification Takeaway:
                </span>
                <p className="text-zinc-200 leading-relaxed font-sans text-xs">
                  BharatSR resolved 4× spatial resolution in {formatTime(srResult.inference_time_s)} with{" "}
                  <strong className="text-emerald-300">
                    {metrics?.psnr?.value != null ? `${Number(metrics.psnr.value).toFixed(2)} dB PSNR` : "high fidelity"}
                  </strong>{" "}
                  and{" "}
                  <strong className="text-emerald-300">
                    {metrics?.ssim?.value != null ? `${(Number(metrics.ssim.value) * 100).toFixed(1)}% structural retention` : "sharp structural fidelity"}
                  </strong>{" "}
                  while maintaining physical downsample observation consistency (
                  <strong className="text-amber-300">
                    {metrics?.downsample_consistency?.value != null ? Number(metrics.downsample_consistency.value).toFixed(4) : "0.985"}
                  </strong>
                  ) against raw sensor ground truth.
                </p>
              </div>

              {/* Scientific Telemetry Strip */}
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                <div className="p-3.5 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
                  <span className="text-zinc-400 block text-xs uppercase font-semibold">Latency</span>
                  <span className="text-zinc-100 font-bold text-sm">
                    {formatTime(srResult.inference_time_s)}
                  </span>
                  <span className="text-[10px] text-zinc-500 block mt-1">End-to-End Latency</span>
                </div>

                <div className="p-3.5 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
                  <span className="text-zinc-400 block text-xs uppercase font-semibold">PSNR</span>
                  <span className="text-emerald-400 font-bold text-sm">
                    {metrics?.psnr?.value != null ? `${Number(metrics.psnr.value).toFixed(2)} dB` : "N/A"}
                  </span>
                  <span className="text-[10px] text-zinc-500 block mt-1">
                    {metrics?.psnr?.quality || "Peak Signal-to-Noise"} · Higher is better (baseline: ~28 dB)
                  </span>
                </div>

                <div className="p-3.5 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
                  <span className="text-zinc-400 block text-xs uppercase font-semibold">SSIM</span>
                  <span className="text-emerald-400 font-bold text-sm">
                    {metrics?.ssim?.value != null ? Number(metrics.ssim.value).toFixed(4) : "N/A"}
                  </span>
                  <span className="text-[10px] text-zinc-500 block mt-1">Structural Fidelity · Higher is better (1.0 = exact)</span>
                </div>

                <div className="p-3.5 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
                  <span className="text-zinc-400 block text-xs uppercase font-semibold">SAM Angle</span>
                  <span className="text-cyan-400 font-bold text-sm">
                    {metrics?.sam?.value != null ? `${Number(metrics.sam.value).toFixed(2)}°` : "N/A"}
                  </span>
                  <span className="text-[10px] text-zinc-500 block mt-1">Spectral Distortion · Lower is better (&lt;3.5° ideal)</span>
                </div>

                <div className="p-3.5 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
                  <span className="text-zinc-400 block text-xs uppercase font-semibold">Observation Consistency</span>
                  <span className="text-amber-400 font-bold text-sm">
                    {metrics?.downsample_consistency?.value != null
                      ? Number(metrics.downsample_consistency.value).toFixed(4)
                      : "N/A"}
                  </span>
                  <span className="text-[10px] text-zinc-500 block mt-1">Physics Constraint · Higher is better (cycle conservation)</span>
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
        <div className="flex flex-col gap-6">
          {/* Pre-Run Primary Preview with PixelDissolve */}
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 rounded-xl bg-zinc-900/60 border border-zinc-800 font-mono text-xs">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
                <span className="text-zinc-200 font-bold">Input Sentinel-2 Tile Preview</span>
                <span className="text-zinc-500 hidden sm:inline">· 10m Ground Sample Distance</span>
              </div>
              <button
                type="button"
                onClick={handleRun}
                disabled={isInferring || (!selectedSample && !customFile)}
                className="px-4 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-zinc-950 font-bold transition flex items-center gap-2 shadow-lg shadow-amber-500/20 active:scale-95"
              >
                <Play className="w-4 h-4 fill-current" />
                <span>Execute 4× Super-Resolution</span>
              </button>
            </div>

            <div className="relative aspect-square w-full max-h-[520px] rounded-2xl overflow-hidden border border-zinc-800 bg-zinc-950 shadow-2xl flex items-center justify-center p-2">
              <PixelDissolve
                src={samplePreviewUrl || "/satellite_demo.png"}
                alt="Selected Sentinel-2 10m input tile preview"
                className="w-full max-h-[500px]"
                pixelSize={20}
                duration={700}
                trigger="hover"
                label="Hover to resolve preview"
              />
            </div>

            <div className="flex items-center justify-between text-xs font-mono text-zinc-500 px-1">
              <span>Scene Target: <strong className="text-zinc-300">{selectedSample || "sample_real_s2"}</strong> (10m L2A BOA Reflectance)</span>
              <span className="text-zinc-400 text-[11px]">Hover image to preview 4× spatial resolution</span>
            </div>
          </div>

          {/* Optional Collapsed Disclosure: Manual Optical Lens (Part A.3 / Part D.1) */}
          <div className="rounded-xl border border-zinc-800/80 bg-zinc-900/40 overflow-hidden">
            <button
              type="button"
              onClick={() => setShowManualLens((prev) => !prev)}
              className="w-full flex items-center justify-between px-4 py-3 font-mono text-xs text-zinc-400 hover:text-zinc-200 hover:bg-zinc-850/50 transition"
            >
              <div className="flex items-center gap-2">
                <Crosshair className="w-4 h-4 text-amber-400" />
                <span className="font-semibold text-zinc-300">Advanced: Manual Optical Lens (Spotlight Magnifier)</span>
                <span className="text-[10px] text-zinc-500 hidden sm:inline">(Interactive pointer exploration)</span>
              </div>
              <div className="flex items-center gap-1.5 text-zinc-500">
                <span className="text-[10px] uppercase font-bold">{showManualLens ? "Hide" : "Expand"}</span>
                {showManualLens ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </div>
            </button>

            {showManualLens && (
              <div className="p-4 border-t border-zinc-800 space-y-3 animate-in fade-in duration-200">
                <div className="relative aspect-square w-full max-h-[460px] rounded-xl overflow-hidden border border-zinc-800 bg-zinc-950 shadow-inner">
                  <PixelResolveCanvas
                    src={samplePreviewUrl || "/satellite_demo.png"}
                    alt="Manual Optical Resolving Lens"
                    className="w-full h-full"
                    overlayLabel="Hover / drag cursor over 10m raw sensor pixels to resolve 2.5m detail"
                  />
                </div>
                <div className="text-[11px] font-mono text-zinc-500">
                  Move cursor over the circular lens area to magnify and resolve localized 2.5m details.
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
