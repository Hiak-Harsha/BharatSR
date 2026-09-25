"use client";

import React, { useState, useEffect } from "react";
import { useChangeDetection } from "../hooks/useChangeDetection";
import { useSamples } from "@/features/samples/hooks/useSamples";
import { useConsoleStore } from "@/lib/store";
import { History, Play, Activity, AlertCircle, ShieldAlert } from "lucide-react";
import { cn } from "@/lib/utils";

export function ChangeDetectionPanel({ className }: { className?: string }) {
  const currentRunId = useConsoleStore((s) => s.currentRunId);
  const { data: samples } = useSamples();

  const [runIdT1, setRunIdT1] = useState<string>("sample_1");
  const [runIdT2, setRunIdT2] = useState<string>("sample_2");
  const [method, setMethod] = useState<string>("ndvi_diff");

  // Automatically update T2 if a fresh run is executed in console
  useEffect(() => {
    if (currentRunId && currentRunId !== runIdT1) {
      setRunIdT2(currentRunId);
    }
  }, [currentRunId, runIdT1]);

  const {
    mutate: runChangeDetect,
    data: changeData,
    isPending: isDetecting,
    error: changeError,
  } = useChangeDetection();

  const handleDetect = () => {
    runChangeDetect({
      runIdT1,
      runIdT2,
      method,
    });
  };

  const stats = (changeData?.statistics || {}) as Record<string, any>;

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl border border-zinc-800 bg-zinc-950">
        <div>
          <h2 className="font-mono text-base font-bold text-zinc-100 flex items-center gap-2">
            <History className="w-5 h-5 text-amber-400" />
            Bi-Temporal Satellite Change Detection
          </h2>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            Compare earlier (T1) vs later (T2) acquisitions to detect land-cover transition or vegetation loss
          </p>
        </div>

        <button
          type="button"
          disabled={isDetecting || !runIdT1 || !runIdT2}
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
              Computing Bi-Temporal Differential...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              Execute Change Detection
            </>
          )}
        </button>
      </div>

      {/* Target Runs Selector */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 p-4 rounded-xl border border-zinc-800 bg-zinc-950 font-mono text-xs">
        <div>
          <label className="text-zinc-500 block text-[11px] uppercase mb-1">
            Reference Epoch (T1 - Earlier)
          </label>
          <select
            value={runIdT1}
            onChange={(e) => setRunIdT1(e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-800 px-3 py-2 rounded-lg text-zinc-200 focus:border-amber-500 focus:outline-none cursor-pointer"
          >
            {samples?.map((s) => (
              <option key={s.id} value={s.id}>
                {s.title} ({s.id})
              </option>
            ))}
            {currentRunId && (
              <option value={currentRunId}>Active Run ({currentRunId})</option>
            )}
            <option value="custom">Custom ID / Manual Entry</option>
          </select>
          {runIdT1 === "custom" && (
            <input
              type="text"
              onChange={(e) => setRunIdT1(e.target.value)}
              placeholder="Enter run_id or sample_id"
              className="mt-2 w-full bg-zinc-900 border border-zinc-800 px-3 py-1.5 rounded text-zinc-200 text-xs focus:border-amber-500 focus:outline-none"
            />
          )}
        </div>

        <div>
          <label className="text-zinc-500 block text-[11px] uppercase mb-1">
            Target Epoch (T2 - Later)
          </label>
          <select
            value={runIdT2}
            onChange={(e) => setRunIdT2(e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-800 px-3 py-2 rounded-lg text-zinc-200 focus:border-amber-500 focus:outline-none cursor-pointer"
          >
            {currentRunId && (
              <option value={currentRunId}>Active Run ({currentRunId})</option>
            )}
            {samples?.map((s) => (
              <option key={s.id} value={s.id}>
                {s.title} ({s.id})
              </option>
            ))}
            <option value="custom">Custom ID / Manual Entry</option>
          </select>
          {runIdT2 === "custom" && (
            <input
              type="text"
              onChange={(e) => setRunIdT2(e.target.value)}
              placeholder="Enter run_id or sample_id"
              className="mt-2 w-full bg-zinc-900 border border-zinc-800 px-3 py-1.5 rounded text-zinc-200 text-xs focus:border-amber-500 focus:outline-none"
            />
          )}
        </div>

        <div>
          <label className="text-zinc-500 block text-[11px] uppercase mb-1">
            Differential Operator
          </label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-800 px-3 py-2 rounded-lg text-amber-300 focus:border-amber-500 focus:outline-none cursor-pointer"
          >
            <option value="ndvi_diff">Normalized Difference Vegetation Index (NDVI)</option>
            <option value="spectral_diff">Multi-Spectral Euclidean Magnitude</option>
            <option value="magnitude">Ratio of Absolute Radiance Change</option>
          </select>
        </div>
      </div>

      {changeError && (
        <div className="p-4 rounded-xl border border-rose-800/60 bg-rose-950/20 text-rose-300 font-mono text-xs flex items-start gap-3">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-400 mt-0.5" />
          <div>
            <span className="font-bold text-rose-200">Change Detection Failed:</span>
            <p className="mt-1">{changeError.message}</p>
          </div>
        </div>
      )}

      {isDetecting ? (
        <div className="aspect-video w-full rounded-xl border border-zinc-800 bg-zinc-950 flex flex-col items-center justify-center p-8 gap-4">
          <div className="w-12 h-12 rounded-full border-2 border-amber-500/20 border-t-amber-400 animate-spin" />
          <div className="font-mono text-xs text-zinc-400 text-center">
            Aligning multi-temporal spatial grids and computing radiometric divergence...
          </div>
        </div>
      ) : changeData ? (
        <div className="flex flex-col gap-6">
          {/* Maps Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-3 flex flex-col gap-2">
              <span className="font-mono text-xs font-semibold text-zinc-300">
                NDVI Differential Map
              </span>
              <div className="aspect-square w-full rounded bg-black overflow-hidden border border-zinc-800">
                <img
                  src={changeData.ndvi_difference_map}
                  alt="NDVI Diff"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>

            <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-3 flex flex-col gap-2">
              <span className="font-mono text-xs font-semibold text-zinc-300">
                Spectral Vector Diff
              </span>
              <div className="aspect-square w-full rounded bg-black overflow-hidden border border-zinc-800">
                <img
                  src={changeData.spectral_difference_map}
                  alt="Spectral Diff"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>

            <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-3 flex flex-col gap-2">
              <span className="font-mono text-xs font-semibold text-zinc-300">
                Change Magnitude (Thresholded)
              </span>
              <div className="aspect-square w-full rounded bg-black overflow-hidden border border-zinc-800">
                <img
                  src={changeData.change_magnitude_map}
                  alt="Magnitude"
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          </div>

          {/* Statistics Strip */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-500 text-[10px] block uppercase">Significant Change</span>
              <span className="text-amber-400 font-bold text-base">
                {stats.significant_change_pct != null
                  ? `${stats.significant_change_pct.toFixed(2)}%`
                  : "N/A"}
              </span>
              <span className="text-[10px] text-zinc-500 block mt-0.5">Thresholded Area</span>
            </div>

            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-500 text-[10px] block uppercase">NDVI Delta (T2 - T1)</span>
              <span className="text-cyan-400 font-bold text-base">
                {stats.ndvi_change != null ? stats.ndvi_change.toFixed(3) : "N/A"}
              </span>
              <span className="text-[10px] text-zinc-500 block mt-0.5">Mean Difference</span>
            </div>

            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-500 text-[10px] block uppercase">Mean Spectral Diff</span>
              <span className="text-zinc-200 font-bold text-base">
                {stats.mean_spectral_diff != null ? stats.mean_spectral_diff.toFixed(4) : "N/A"}
              </span>
              <span className="text-[10px] text-zinc-500 block mt-0.5">Euclidean Norm</span>
            </div>

            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-500 text-[10px] block uppercase">Max Magnitude</span>
              <span className="text-zinc-200 font-bold text-base">
                {stats.max_change_magnitude != null ? stats.max_change_magnitude.toFixed(3) : "N/A"}
              </span>
              <span className="text-[10px] text-zinc-500 block mt-0.5">Peak Anomaly</span>
            </div>
          </div>

          {changeData.interpretation && (
            <div className="p-3.5 rounded-xl border border-zinc-800 bg-zinc-950 font-mono text-xs text-zinc-300">
              <span className="font-bold text-amber-300 uppercase block mb-1">
                Automated Analytical Interpretation:
              </span>
              <p>{changeData.interpretation}</p>
            </div>
          )}

          {changeData.disclaimer && (
            <div className="rounded-lg bg-amber-950/20 border border-amber-800/40 p-2.5 text-xs font-mono text-amber-300/80 flex items-start gap-2">
              <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <span>{changeData.disclaimer}</span>
            </div>
          )}
        </div>
      ) : (
        <div className="p-12 rounded-xl border border-dashed border-zinc-800 bg-zinc-950/40 text-center font-mono text-xs text-zinc-500 flex flex-col items-center gap-3">
          <History className="w-8 h-8 text-zinc-700" />
          <span>Select T1 and T2 from samples or active runs, then click "Execute Change Detection".</span>
        </div>
      )}
    </div>
  );
}
