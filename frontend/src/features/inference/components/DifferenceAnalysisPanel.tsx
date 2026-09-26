"use client";

import React, { useState, useRef } from "react";
import { cn } from "@/lib/utils";
import { UncertaintyScatterChart } from "@/components/charts/UncertaintyScatterChart";
import { Activity, Eye, Info, Crosshair, ShieldAlert } from "lucide-react";

export interface DifferenceAnalysisPanelProps {
  errorMap?: {
    image?: string;
    image_raw?: string;
    image_heatmap?: string;
    mean_error?: number;
    max_error?: number;
    min_error?: number;
    normalization?: string;
  };
  uncertainty?: {
    image?: string;
    summary?: any;
    scatter?: any;
  };
  hasGroundTruth?: boolean;
  onInspectPixel?: (x: number, y: number) => void;
  inspectedPoint?: { x: number; y: number } | null;
  imageDimensions?: { width: number; height: number };
  className?: string;
}

export function DifferenceAnalysisPanel({
  errorMap,
  uncertainty,
  hasGroundTruth = true,
  onInspectPixel,
  inspectedPoint,
  imageDimensions = { width: 256, height: 256 },
  className,
}: DifferenceAnalysisPanelProps) {
  const [activeTab, setActiveTab] = useState<"raw" | "heatmap" | "uncertainty">("raw");
  const imgRef = useRef<HTMLImageElement>(null);

  const handleImageClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!onInspectPixel || !imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const normX = Math.max(0, Math.min(1, clickX / rect.width));
    const normY = Math.max(0, Math.min(1, clickY / rect.height));
    const pixelX = Math.round(normX * (imageDimensions.width - 1));
    const pixelY = Math.round(normY * (imageDimensions.height - 1));
    onInspectPixel(pixelX, pixelY);
  };

  const rawImage = errorMap?.image_raw || errorMap?.image;
  const heatmapImage = errorMap?.image_heatmap || errorMap?.image;
  const uncertaintyImage = uncertainty?.image;

  const minErr = errorMap?.min_error ?? 0.0;
  const maxErr = errorMap?.max_error ?? 0.08;
  const meanErr = errorMap?.mean_error ?? 0.015;

  const summary = uncertainty?.summary || {};
  const minUnc = summary.min_uncertainty ?? summary.min_sigma ?? 0.0;
  const maxUnc = summary.max_uncertainty ?? summary.max_sigma ?? 0.12;
  const meanUnc = summary.mean_uncertainty ?? summary.mean_sigma ?? 0.03;

  return (
    <div className={cn("rounded-2xl border border-zinc-800 bg-zinc-950 p-6 flex flex-col gap-6", className)}>
      {/* Header and Navigable Tabs */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-zinc-850">
        <div>
          <h3 className="font-mono text-sm font-bold text-zinc-100 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            Difference &amp; Uncertainty Analysis
          </h3>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            Modular diagnostics: uncolored magnitude difference, analytical error heatmap, and Bayesian uncertainty.
          </p>
        </div>

        {/* Tab Segmented Control */}
        <div className="flex items-center gap-1 p-1 bg-zinc-900 border border-zinc-800 rounded-xl font-mono text-xs self-start sm:self-auto">
          <button
            type="button"
            onClick={() => setActiveTab("raw")}
            className={cn(
              "px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5",
              activeTab === "raw"
                ? "bg-zinc-800 text-amber-400 shadow"
                : "text-zinc-400 hover:text-zinc-200"
            )}
          >
            <Eye className="w-3.5 h-3.5" />
            Raw Difference
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("heatmap")}
            className={cn(
              "px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5",
              activeTab === "heatmap"
                ? "bg-zinc-800 text-amber-400 shadow"
                : "text-zinc-400 hover:text-zinc-200"
            )}
          >
            Error Heatmap
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("uncertainty")}
            className={cn(
              "px-3 py-1.5 rounded-lg text-xs font-semibold transition flex items-center gap-1.5",
              activeTab === "uncertainty"
                ? "bg-zinc-800 text-amber-400 shadow"
                : "text-zinc-400 hover:text-zinc-200"
            )}
          >
            Predicted Uncertainty
          </button>
        </div>
      </div>

      {/* TAB 1: RAW DIFFERENCE (GRAYSCALE, UNCOLORED) */}
      {activeTab === "raw" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Main Visual Display */}
          <div className="lg:col-span-7 flex flex-col gap-3">
            <div
              className="relative aspect-square w-full rounded-xl overflow-hidden border border-zinc-800 bg-black cursor-crosshair group"
              onClick={handleImageClick}
            >
              {rawImage ? (
                <img
                  ref={imgRef}
                  src={rawImage}
                  alt="Raw Grayscale Difference Map"
                  className="w-full h-full object-contain pixelated"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center font-mono text-xs text-zinc-500">
                  No difference map available for this run.
                </div>
              )}

              {/* Inspected coordinate crosshair overlay */}
              {inspectedPoint && (
                <div
                  className="absolute pointer-events-none -translate-x-1/2 -translate-y-1/2 z-10"
                  style={{
                    left: `${(inspectedPoint.x / (imageDimensions.width - 1)) * 100}%`,
                    top: `${(inspectedPoint.y / (imageDimensions.height - 1)) * 100}%`,
                  }}
                >
                  <div className="w-6 h-6 rounded-full border-2 border-amber-400/90 shadow-[0_0_8px_rgba(245,158,11,0.8)] flex items-center justify-center animate-pulse">
                    <div className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                  </div>
                </div>
              )}

              <div className="absolute bottom-2 left-2 px-2 py-1 rounded bg-black/80 backdrop-blur border border-zinc-800 text-[11px] font-mono text-zinc-300 flex items-center gap-1.5">
                <Crosshair className="w-3 h-3 text-amber-400" />
                Click anywhere to inspect multi-band spectral reflectance
              </div>
            </div>

            {/* Uncolored Magnitude Explanation */}
            <div className="p-3.5 rounded-xl border border-zinc-800/80 bg-zinc-900/40 text-xs font-mono text-zinc-300 flex items-start gap-2.5">
              <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-zinc-100">Raw Radiometric Magnitude (Uncolored):</span>
                <p className="mt-0.5 text-zinc-400 leading-relaxed font-sans text-xs">
                  Brighter pixels = larger absolute deviation from reference (<code className="font-mono text-zinc-300">|SR - Reference|</code>).
                  Uncolored single-channel image representing true physical magnitude with no false-color distortion.
                </p>
              </div>
            </div>
          </div>

          {/* Quantitative Metrics & Telemetry */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-900/60 font-mono space-y-3">
              <span className="text-zinc-400 text-xs font-bold uppercase tracking-wider block">
                Reconstruction Deviation Statistics
              </span>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                  <span className="text-zinc-500 text-[11px] block">Mean Absolute Error (L1)</span>
                  <span className="text-amber-400 font-bold text-base mt-0.5 block">
                    {meanErr.toFixed(5)}
                  </span>
                  <span className="text-[10px] text-zinc-500 mt-1 block">Reflectance units [0-1]</span>
                </div>

                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                  <span className="text-zinc-500 text-[11px] block">Peak Error (Max L1)</span>
                  <span className="text-rose-400 font-bold text-base mt-0.5 block">
                    {maxErr.toFixed(5)}
                  </span>
                  <span className="text-[10px] text-zinc-500 mt-1 block">Reflectance units [0-1]</span>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 text-xs text-zinc-400">
                <span className="text-zinc-500 text-[10px] uppercase block font-semibold">Image Normalization</span>
                <p className="text-zinc-300 text-[11px] mt-1 font-mono">
                  {errorMap?.normalization || `Linear 0.0 to ${maxErr.toFixed(5)} stretched to [0, 255] grayscale`}
                </p>
              </div>
            </div>

            <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-900/40 text-xs font-mono text-zinc-400 space-y-2">
              <span className="text-zinc-300 font-semibold uppercase text-[11px] block">
                Verification Integrity Standard
              </span>
              <p className="font-sans text-xs text-zinc-400 leading-relaxed">
                In scientific satellite imaging, synthetic colorization can artificially amplify minor noise.
                BharatSR provides this uncalibrated raw difference map to enable defensible validation against native ground sensors.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: ERROR HEATMAP (PLASMA COLORIZED) */}
      {activeTab === "heatmap" && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          <div className="lg:col-span-7 flex flex-col gap-3">
            <div
              className="relative aspect-square w-full rounded-xl overflow-hidden border border-zinc-800 bg-black cursor-crosshair"
              onClick={handleImageClick}
            >
              {heatmapImage ? (
                <img
                  ref={imgRef}
                  src={heatmapImage}
                  alt="Error Heatmap (Plasma Colormap)"
                  className="w-full h-full object-contain pixelated"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center font-mono text-xs text-zinc-500">
                  No error heatmap available.
                </div>
              )}

              {inspectedPoint && (
                <div
                  className="absolute pointer-events-none -translate-x-1/2 -translate-y-1/2 z-10"
                  style={{
                    left: `${(inspectedPoint.x / (imageDimensions.width - 1)) * 100}%`,
                    top: `${(inspectedPoint.y / (imageDimensions.height - 1)) * 100}%`,
                  }}
                >
                  <div className="w-6 h-6 rounded-full border-2 border-amber-400/90 shadow-[0_0_8px_rgba(245,158,11,0.8)] flex items-center justify-center animate-pulse">
                    <div className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                  </div>
                </div>
              )}
            </div>

            {/* Run-Specific Color Scale Legend */}
            <div className="p-3.5 rounded-xl border border-zinc-800 bg-zinc-900/60 flex flex-col gap-2 font-mono">
              <div className="flex justify-between items-center text-xs">
                <span className="text-zinc-400 font-semibold">Perceptual Plasma Colormap Scale</span>
                <span className="text-[11px] text-zinc-500">Run-Specific Bounds</span>
              </div>
              <div
                className="h-3 w-full rounded-md shadow-inner border border-zinc-700/60"
                style={{
                  background: "linear-gradient(to right, #0d0887, #6a00a8, #b12a90, #e16462, #fca636, #f0f921)",
                }}
              />
              <div className="flex justify-between items-center text-[11px] text-zinc-400">
                <span>Min: {minErr.toFixed(5)}</span>
                <span className="text-amber-400">Mean: {meanErr.toFixed(5)}</span>
                <span>Max: {maxErr.toFixed(5)}</span>
              </div>
            </div>
          </div>

          <div className="lg:col-span-5 flex flex-col gap-4">
            <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-900/60 font-mono space-y-3">
              <span className="text-zinc-400 text-xs font-bold uppercase tracking-wider block">
                Visualization Analytical Add-On
              </span>
              <p className="font-sans text-xs text-zinc-300 leading-relaxed">
                The Plasma colormap assigns dark purple to zero deviation and bright yellow to maximum observed error.
                This enhances structural edge transitions (such as crop boundaries and roads) where high-frequency hallucination might occur.
              </p>
              <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 text-xs">
                <span className="text-zinc-500 text-[10px] uppercase block font-semibold">Spectral Bands Evaluated</span>
                <span className="text-zinc-200 font-bold block mt-0.5">Mean absolute error averaged across B2, B3, B4, B8</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: PREDICTED UNCERTAINTY (MAGMA COLORIZED & SCATTER) */}
      {activeTab === "uncertainty" && (
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            <div className="lg:col-span-7 flex flex-col gap-3">
              <div
                className="relative aspect-square w-full rounded-xl overflow-hidden border border-zinc-800 bg-black cursor-crosshair"
                onClick={handleImageClick}
              >
                {uncertaintyImage ? (
                  <img
                    ref={imgRef}
                    src={uncertaintyImage}
                    alt="Bayesian Uncertainty Map (Magma Colormap)"
                    className="w-full h-full object-contain pixelated"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center font-mono text-xs text-zinc-500">
                    No uncertainty estimation available for this model architecture.
                  </div>
                )}

                {inspectedPoint && (
                  <div
                    className="absolute pointer-events-none -translate-x-1/2 -translate-y-1/2 z-10"
                    style={{
                      left: `${(inspectedPoint.x / (imageDimensions.width - 1)) * 100}%`,
                      top: `${(inspectedPoint.y / (imageDimensions.height - 1)) * 100}%`,
                    }}
                  >
                    <div className="w-6 h-6 rounded-full border-2 border-amber-400/90 shadow-[0_0_8px_rgba(245,158,11,0.8)] flex items-center justify-center animate-pulse">
                      <div className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                    </div>
                  </div>
                )}
              </div>

              {/* Magma Legend */}
              <div className="p-3.5 rounded-xl border border-zinc-800 bg-zinc-900/60 flex flex-col gap-2 font-mono">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-zinc-400 font-semibold">Magma Colormap Scale (Predicted σ)</span>
                  <span className="text-[11px] text-zinc-500">Actual Run Range</span>
                </div>
                <div
                  className="h-3 w-full rounded-md shadow-inner border border-zinc-700/60"
                  style={{
                    background: "linear-gradient(to right, #000004, #3b0f70, #8c2981, #de4968, #fe9f6d, #fcfdbf)",
                  }}
                />
                <div className="flex justify-between items-center text-[11px] text-zinc-400">
                  <span>Min σ: {Number(minUnc).toFixed(4)}</span>
                  <span className="text-amber-400">Mean σ: {Number(meanUnc).toFixed(4)}</span>
                  <span>Max σ: {Number(maxUnc).toFixed(4)}</span>
                </div>
              </div>
            </div>

            {/* Uncertainty Telemetry Summary */}
            <div className="lg:col-span-5 flex flex-col gap-4">
              <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-900/60 font-mono space-y-3">
                <span className="text-zinc-400 text-xs font-bold uppercase tracking-wider block">
                  Heteroscedastic Uncertainty Summary
                </span>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                    <span className="text-zinc-500 text-[11px] block">Mean σ (Standard Dev)</span>
                    <span className="text-zinc-200 font-bold text-base mt-0.5 block">
                      {Number(meanUnc).toFixed(4)}
                    </span>
                  </div>

                  <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                    <span className="text-zinc-500 text-[11px] block">Peak σ (Max Uncertainty)</span>
                    <span className="text-amber-400 font-bold text-base mt-0.5 block">
                      {Number(maxUnc).toFixed(4)}
                    </span>
                  </div>
                </div>

                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 text-xs">
                  <span className="text-zinc-500 text-[10px] uppercase block font-semibold">High Uncertainty Area</span>
                  <span className="text-emerald-400 font-bold text-sm block mt-0.5">
                    {summary.high_uncertainty_fraction != null
                      ? `${(Number(summary.high_uncertainty_fraction) * 100).toFixed(1)}% of scene`
                      : summary.high_uncertainty_pixel_pct != null
                      ? `${Number(summary.high_uncertainty_pixel_pct).toFixed(1)}% of scene`
                      : "0.0%"}
                  </span>
                  <span className="text-[10px] text-zinc-500 block mt-0.5">Pixels exceeding 95th percentile threshold</span>
                </div>
              </div>

              <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-900/40 text-xs font-mono text-zinc-400">
                <span className="text-zinc-300 font-semibold uppercase text-[11px] block mb-1">
                  How Uncertainty Informs Analytics
                </span>
                <p className="font-sans text-xs text-zinc-400 leading-relaxed">
                  Pixels with high predicted σ flag complex, fine textures (e.g. sharp field boundaries or urban edges)
                  where downstream analytics should verify against multi-temporal passes.
                </p>
              </div>
            </div>
          </div>

          {/* Uncertainty Calibration Scatter Section */}
          <div className="border-t border-zinc-850 pt-4">
            {hasGroundTruth ? (
              <UncertaintyScatterChart
                scatter={uncertainty?.scatter}
                summary={uncertainty?.summary}
              />
            ) : (
              <div className="p-5 rounded-xl border border-zinc-800 bg-zinc-950/60 font-mono text-xs flex items-start gap-3 text-zinc-400">
                <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-zinc-200 block text-sm">
                    Reference Ground Truth Required for Error Correlation
                  </span>
                  <p className="mt-1 text-zinc-400 font-sans text-xs leading-relaxed">
                    Uncertainty-vs-error correlation (Pearson r) requires a registered high-resolution reference acquisition to compute actual error (|SR - GT|).
                    When processing user-uploaded or unreferenced Sentinel-2 scenes, predicted σ remains active for confidence filtering, but empirical correlation scatter cannot be calculated.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
