"use client";

import React, { useState } from "react";
import { useFieldBoundary } from "../hooks/useFieldBoundary";
import { useConsoleStore } from "@/lib/store";
import { ImageComparisonSlider } from "@/components/ui/ImageComparisonSlider";
import { Maximize2, Play, Activity, AlertCircle, ShieldAlert } from "lucide-react";
import { cn } from "@/lib/utils";

export function FieldBoundaryOverlay({ className }: { className?: string }) {
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const selectedModel = useConsoleStore((s) => s.selectedModel);
  const currentRunId = useConsoleStore((s) => s.currentRunId);

  const [method, setMethod] = useState<string>("gradient");

  const {
    mutate: runFieldBoundary,
    data: boundaryData,
    isPending: isDetecting,
    error: boundaryError,
  } = useFieldBoundary();

  const handleDetect = () => {
    runFieldBoundary({
      sampleId: selectedSample || undefined,
      runId: currentRunId || undefined,
      modelId: selectedModel,
      method,
    });
  };

  const layers = boundaryData
    ? [
        {
          id: "lr-edge",
          label: "LR Boundary Overlay (10m)",
          tag: "Coarse",
          image: boundaryData.lr_edge_overlay,
        },
        {
          id: "sr-edge",
          label: "BharatSR Boundary Overlay (2.5m)",
          tag: "Sharpened",
          image: boundaryData.sr_edge_overlay,
        },
      ]
    : [];

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-950">
        <div>
          <h2 className="font-mono text-base font-bold text-zinc-100 flex items-center gap-2">
            <Maximize2 className="w-5 h-5 text-amber-400" />
            Cadastral & Agricultural Field Boundary Delineation
          </h2>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            Delineate agricultural plots and infrastructure parcel edges using spatial gradient filters
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 bg-zinc-900 border border-zinc-800 px-2.5 py-1.5 rounded-lg text-xs font-mono">
            <span className="text-zinc-500 uppercase">Method:</span>
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value)}
              className="bg-transparent text-amber-300 font-semibold focus:outline-none cursor-pointer"
            >
              <option value="gradient" className="bg-zinc-900">
                Directional Gradient
              </option>
              <option value="laplacian" className="bg-zinc-900">
                Laplacian of Gaussian
              </option>
            </select>
          </div>

          <button
            type="button"
            disabled={isDetecting}
            onClick={handleDetect}
            className={cn(
              "inline-flex items-center gap-2 px-6 py-2 rounded-lg font-mono text-xs font-bold tracking-wider uppercase transition shadow-lg",
              isDetecting
                ? "bg-amber-500/50 text-zinc-950 cursor-not-allowed"
                : "bg-amber-500 hover:bg-amber-400 text-zinc-950 shadow-[0_0_20px_rgba(245,158,11,0.3)] active:scale-95"
            )}
          >
            {isDetecting ? (
              <>
                <Activity className="w-4 h-4 animate-spin" />
                Delineating Boundaries...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Delineate Boundaries
              </>
            )}
          </button>
        </div>
      </div>

      {boundaryError && (
        <div className="p-4 rounded-xl border border-rose-800/60 bg-rose-950/20 text-rose-300 font-mono text-xs flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 mt-0.5" />
          <div>
            <span className="font-bold text-rose-200">Delineation Error:</span>
            <p className="mt-1">{boundaryError.message}</p>
          </div>
        </div>
      )}

      {isDetecting ? (
        <div className="aspect-square w-full max-h-[460px] rounded-xl border border-zinc-800 bg-zinc-950 flex flex-col items-center justify-center p-8 gap-4">
          <div className="w-12 h-12 rounded-full border-2 border-amber-500/20 border-t-amber-400 animate-spin" />
          <div className="font-mono text-xs text-zinc-400 text-center">
            Computing high-frequency spatial derivatives and edge non-maximum suppression...
          </div>
        </div>
      ) : boundaryData ? (
        <>
          <ImageComparisonSlider
            layers={layers}
            initialLeftId="lr-edge"
            initialRightId="sr-edge"
          />

          {/* Delineation Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-500 text-[10px] block uppercase">Boundary Improvement</span>
              <span className="text-emerald-400 font-bold text-base">
                {boundaryData.boundary_improvement_ratio.toFixed(2)}×
              </span>
              <span className="text-[10px] text-zinc-500 block mt-0.5">Sharpening Ratio</span>
            </div>

            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-500 text-[10px] block uppercase">SR Edge Density</span>
              <span className="text-amber-300 font-bold text-base">
                {(boundaryData.sr_edge_density * 100).toFixed(2)}%
              </span>
              <span className="text-[10px] text-zinc-500 block mt-0.5">
                {boundaryData.sr_edge_pixel_count} Pixels
              </span>
            </div>

            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-500 text-[10px] block uppercase">LR Edge Density</span>
              <span className="text-zinc-300 font-bold text-base">
                {(boundaryData.lr_edge_density * 100).toFixed(2)}%
              </span>
              <span className="text-[10px] text-zinc-500 block mt-0.5">
                {boundaryData.lr_edge_pixel_count} Pixels
              </span>
            </div>

            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-500 text-[10px] block uppercase">Operator</span>
              <span className="text-cyan-300 font-bold text-base uppercase">
                {boundaryData.method}
              </span>
              <span className="text-[10px] text-zinc-500 block mt-0.5">Spatial Kernel</span>
            </div>
          </div>

          {boundaryData.disclaimer && (
            <div className="rounded-lg bg-amber-950/20 border border-amber-800/40 p-2.5 text-xs font-mono text-amber-300/80 flex items-start gap-2">
              <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <span>{boundaryData.disclaimer}</span>
            </div>
          )}
        </>
      ) : (
        <div className="p-12 rounded-xl border border-dashed border-zinc-800 bg-zinc-950/40 text-center font-mono text-xs text-zinc-500 flex flex-col items-center gap-3">
          <Maximize2 className="w-8 h-8 text-zinc-700" />
          <span>Click "Delineate Boundaries" to extract field contours and cadastral lines.</span>
        </div>
      )}
    </div>
  );
}
