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
      runId: currentRunId || undefined,
      sampleId: currentRunId ? undefined : (selectedSample || undefined),
      modelId: selectedModel,
      method,
    });
  };

  const layers = boundaryData
    ? [
        {
          id: "lr-edge",
          label: "LR Edge Map (10m Input)",
          tag: "10m Raw",
          image: boundaryData.lr_edge_overlay,
        },
        {
          id: "sr-edge",
          label: "BharatSR Edge Map (2.5m-equivalent SR grid)",
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
            Heuristic Field Boundary Detection
          </h2>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            {currentRunId
              ? `Evaluating spatial gradients on completed run ${currentRunId}`
              : "High-frequency edge sharpness detection across agricultural parcels"}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 bg-zinc-900 border border-zinc-800 px-3 py-1.5 rounded-lg text-xs font-mono">
            <span className="text-zinc-400 uppercase font-semibold">Method:</span>
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
            disabled={isDetecting || (!currentRunId && !selectedSample)}
            onClick={handleDetect}
            className={cn(
              "inline-flex items-center gap-2 px-6 py-2.5 rounded-lg font-mono text-xs font-bold tracking-wider uppercase transition shadow-lg",
              isDetecting
                ? "bg-amber-500/50 text-zinc-950 cursor-not-allowed"
                : "bg-amber-500 hover:bg-amber-400 text-zinc-950 shadow-[0_0_20px_rgba(245,158,11,0.3)] active:scale-95"
            )}
          >
            {isDetecting ? (
              <>
                <Activity className="w-4 h-4 animate-spin" />
                Detecting Edges...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                Detect Field Boundaries
              </>
            )}
          </button>
        </div>
      </div>

      {/* Prominent Mandatory Cadastral Disclaimer */}
      <div className="rounded-lg bg-amber-950/30 border border-amber-800/50 p-3.5 text-xs font-mono text-amber-300 flex items-start gap-2.5">
        <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-amber-200">Cadastral Limitation Disclaimer: </span>
          <span>
            This tool performs mathematical edge filtering on raster gradients. It is a heuristic edge detector and does NOT represent legal, certified, or authoritative cadastral land parcel boundaries.
          </span>
        </div>
      </div>

      {boundaryError && (
        <div className="p-4 rounded-xl border border-rose-800/60 bg-rose-950/20 text-rose-300 font-mono text-xs flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 mt-0.5" />
          <div>
            <span className="font-bold text-rose-200">Detection Error:</span>
            <p className="mt-1">{boundaryError.message}</p>
          </div>
        </div>
      )}

      {isDetecting ? (
        <div className="aspect-square w-full max-h-[460px] rounded-xl border border-zinc-800 bg-zinc-950 flex flex-col items-center justify-center p-8 gap-4">
          <div className="w-12 h-12 rounded-full border-2 border-amber-500/20 border-t-amber-400 animate-spin" />
          <div className="font-mono text-xs text-zinc-300 text-center">
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
            <div className="p-3.5 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-400 text-xs block uppercase font-semibold">Boundary Improvement</span>
              <span className="text-emerald-400 font-bold text-lg">
                {boundaryData.boundary_improvement_ratio.toFixed(2)}×
              </span>
              <span className="text-xs text-zinc-500 block mt-1">Sharpening Ratio</span>
            </div>

            <div className="p-3.5 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-400 text-xs block uppercase font-semibold">SR Edge Density</span>
              <span className="text-amber-300 font-bold text-lg">
                {(boundaryData.sr_edge_density * 100).toFixed(2)}%
              </span>
              <span className="text-xs text-zinc-500 block mt-1">
                {boundaryData.sr_edge_pixel_count} Pixels
              </span>
            </div>

            <div className="p-3.5 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-400 text-xs block uppercase font-semibold">LR Edge Density</span>
              <span className="text-zinc-200 font-bold text-lg">
                {(boundaryData.lr_edge_density * 100).toFixed(2)}%
              </span>
              <span className="text-xs text-zinc-500 block mt-1">
                {boundaryData.lr_edge_pixel_count} Pixels
              </span>
            </div>

            <div className="p-3.5 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-400 text-xs block uppercase font-semibold">Operator</span>
              <span className="text-cyan-300 font-bold text-lg uppercase">
                {boundaryData.method}
              </span>
              <span className="text-xs text-zinc-500 block mt-1">Spatial Kernel</span>
            </div>
          </div>
        </>
      ) : (
        <div className="p-12 rounded-xl border border-dashed border-zinc-800 bg-zinc-950/40 text-center font-mono text-xs text-zinc-400 flex flex-col items-center gap-3">
          <Maximize2 className="w-8 h-8 text-zinc-600" />
          <span className="text-zinc-300 font-medium">
            No field boundary analysis has been generated for the current run.
          </span>
          <button
            type="button"
            onClick={handleDetect}
            disabled={!currentRunId && !selectedSample}
            className="mt-2 px-4 py-2 rounded bg-amber-500 hover:bg-amber-400 text-zinc-950 font-bold transition"
          >
            Detect Field Boundaries
          </button>
        </div>
      )}
    </div>
  );
}
