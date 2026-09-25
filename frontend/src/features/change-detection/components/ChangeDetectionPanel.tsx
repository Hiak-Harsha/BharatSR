"use client";

import React, { useState, useEffect } from "react";
import { useChangeDetection } from "../hooks/useChangeDetection";
import { useSamples } from "@/features/samples/hooks/useSamples";
import { useConsoleStore } from "@/lib/store";
import { History, Play, Activity, AlertCircle, ShieldAlert } from "lucide-react";
import { cn } from "@/lib/utils";

export function ChangeDetectionPanel({ className }: { className?: string }) {
  const currentRunId = useConsoleStore((s) => s.currentRunId);
  const currentSession = useConsoleStore((s) => s.currentSession);
  const { data: samples } = useSamples();

  const [t1Key, setT1Key] = useState<string>("sample_real_s2");
  const [t2Key, setT2Key] = useState<string>("sample_1");
  const [method, setMethod] = useState<string>("ndvi_diff");

  // Automatically update T2 if a fresh run is executed in console
  useEffect(() => {
    if (currentRunId) {
      setT2Key(currentRunId);
    }
  }, [currentRunId]);

  const {
    mutate: runChangeDetect,
    data: changeData,
    isPending: isDetecting,
    error: changeError,
  } = useChangeDetection();

  const handleDetect = () => {
    runChangeDetect({
      runIdT1: t1Key,
      runIdT2: t2Key,
      method,
    });
  };

  const getOptionMetadata = (key: string) => {
    if (key === currentRunId) {
      return {
        title: "Active Console Run",
        date: currentSession?.createdAt ? new Date(currentSession.createdAt).toLocaleDateString() : "Current Session",
        sensor: "Sentinel-2 L2A (Processed)",
        model: currentSession?.modelId?.toUpperCase() || "RCAN",
        status: "Active Run",
        isDemo: false,
      };
    }
    const sample = samples?.find((s) => s.sample_id === key);
    if (sample) {
      return {
        title: sample.title,
        date: sample.acquisition_date || "Demo scene — generated inference",
        sensor: sample.sensor || "Sentinel-2 MSI",
        model: "Original / 4x SR",
        status: sample.is_independent_hr ? "Independent Observation" : "Demonstration Scene",
        isDemo: !sample.acquisition_date,
      };
    }
    return {
      title: key,
      date: "External Artifact",
      sensor: "Custom Raster",
      model: "Unknown",
      status: "Custom",
      isDemo: true,
    };
  };

  const metaT1 = getOptionMetadata(t1Key);
  const metaT2 = getOptionMetadata(t2Key);

  // Check compatibility
  const isSameTarget = t1Key === t2Key;
  const isCompatible = !isSameTarget;

  const stats = (changeData?.statistics || {}) as Record<string, any>;

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      <div className="flex flex-wrap items-center justify-between gap-4 p-5 rounded-xl border border-zinc-800 bg-zinc-950">
        <div>
          <h2 className="font-mono text-base font-bold text-zinc-100 flex items-center gap-2">
            <History className="w-5 h-5 text-amber-400" />
            Bi-Temporal Satellite Change Detection
          </h2>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            Compare earlier (TIME 1) vs later (TIME 2) acquisitions to quantify vegetation delta and radiometric shift.
          </p>
        </div>

        <button
          type="button"
          disabled={isDetecting || !isCompatible}
          onClick={handleDetect}
          className={cn(
            "inline-flex items-center gap-2 px-6 py-2.5 rounded-lg font-mono text-xs font-bold tracking-wider uppercase transition shadow-lg",
            isDetecting || !isCompatible
              ? "bg-zinc-800 text-zinc-500 cursor-not-allowed"
              : "bg-amber-500 hover:bg-amber-400 text-zinc-950 shadow-[0_0_20px_rgba(245,158,11,0.3)] active:scale-95"
          )}
        >
          {isDetecting ? (
            <>
              <Activity className="w-4 h-4 animate-spin" />
              Computing Radiometric Divergence...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-current" />
              Execute Change Detection
            </>
          )}
        </button>
      </div>

      {/* Target Runs Selector: TIME 1 vs TIME 2 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* TIME 1 CARD */}
        <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950 font-mono space-y-3">
          <div className="flex items-center justify-between border-b border-zinc-850 pb-2">
            <span className="text-xs font-bold uppercase tracking-wider text-amber-400">
              TIME 1 (Earlier Epoch)
            </span>
            <span className="text-xs text-zinc-400">{metaT1.status}</span>
          </div>

          <div>
            <label className="text-zinc-400 block text-xs mb-1.5">
              Select Scene / Saved Run
            </label>
            <select
              value={t1Key}
              onChange={(e) => setT1Key(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 px-3 py-2.5 rounded-lg text-zinc-100 text-xs focus:border-amber-500 focus:outline-none cursor-pointer"
            >
              {(samples as any[])?.map((s: any) => (
                <option key={s.sample_id} value={s.sample_id}>
                  {s.title} ({s.sensor})
                </option>
              ))}
              {currentRunId && (
                <option value={currentRunId}>Active Console Run ({currentRunId.slice(0, 10)}...)</option>
              )}
            </select>
          </div>

          <div className="rounded-lg bg-zinc-900/80 border border-zinc-800 p-3 text-xs space-y-1 text-zinc-300">
            <div><span className="text-zinc-400">Scene:</span> <span className="font-semibold">{metaT1.title}</span></div>
            <div><span className="text-zinc-400">Date:</span> {metaT1.date}</div>
            <div><span className="text-zinc-400">Sensor:</span> {metaT1.sensor}</div>
          </div>
        </div>

        {/* TIME 2 CARD */}
        <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950 font-mono space-y-3">
          <div className="flex items-center justify-between border-b border-zinc-850 pb-2">
            <span className="text-xs font-bold uppercase tracking-wider text-cyan-400">
              TIME 2 (Later Epoch)
            </span>
            <span className="text-xs text-zinc-400">{metaT2.status}</span>
          </div>

          <div>
            <label className="text-zinc-400 block text-xs mb-1.5">
              Select Scene / Saved Run
            </label>
            <select
              value={t2Key}
              onChange={(e) => setT2Key(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 px-3 py-2.5 rounded-lg text-zinc-100 text-xs focus:border-amber-500 focus:outline-none cursor-pointer"
            >
              {currentRunId && (
                <option value={currentRunId}>Active Console Run ({currentRunId.slice(0, 10)}...)</option>
              )}
              {(samples as any[])?.map((s: any) => (
                <option key={s.sample_id} value={s.sample_id}>
                  {s.title} ({s.sensor})
                </option>
              ))}
            </select>
          </div>

          <div className="rounded-lg bg-zinc-900/80 border border-zinc-800 p-3 text-xs space-y-1 text-zinc-300">
            <div><span className="text-zinc-400">Scene:</span> <span className="font-semibold">{metaT2.title}</span></div>
            <div><span className="text-zinc-400">Date:</span> {metaT2.date}</div>
            <div><span className="text-zinc-400">Sensor:</span> {metaT2.sensor}</div>
          </div>
        </div>
      </div>

      {/* Operator and Compatibility Strip */}
      <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950 font-mono flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex-1">
          <label className="text-zinc-400 block text-xs uppercase mb-1">
            Differential Operator
          </label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value)}
            className="w-full sm:w-80 bg-zinc-900 border border-zinc-800 px-3 py-2 rounded-lg text-amber-300 text-xs focus:border-amber-500 focus:outline-none cursor-pointer"
          >
            <option value="ndvi_diff">Normalized Difference Vegetation Index (NDVI)</option>
            <option value="spectral_diff">Multi-Spectral Euclidean Magnitude</option>
            <option value="magnitude">Ratio of Absolute Radiance Change</option>
          </select>
        </div>

        {isSameTarget && (
          <div className="rounded-lg bg-amber-950/20 border border-amber-800/40 p-2.5 text-xs text-amber-300 flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-amber-400" />
            <span>Select two distinct scenes or runs to evaluate temporal change.</span>
          </div>
        )}
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
              <span className="text-zinc-400 text-xs block uppercase">Significant Change</span>
              <span className="text-amber-400 font-bold text-lg">
                {stats.significant_change_pct != null
                  ? `${stats.significant_change_pct.toFixed(2)}%`
                  : "N/A"}
              </span>
              <span className="text-xs text-zinc-500 block mt-0.5">Thresholded Area</span>
            </div>

            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-400 text-xs block uppercase">NDVI Delta (T2 - T1)</span>
              <span className="text-cyan-400 font-bold text-lg">
                {stats.ndvi_change != null ? stats.ndvi_change.toFixed(3) : "N/A"}
              </span>
              <span className="text-xs text-zinc-500 block mt-0.5">Mean Difference</span>
            </div>

            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-400 text-xs block uppercase">Mean Spectral Diff</span>
              <span className="text-zinc-200 font-bold text-lg">
                {stats.mean_spectral_diff != null ? stats.mean_spectral_diff.toFixed(4) : "N/A"}
              </span>
              <span className="text-xs text-zinc-500 block mt-0.5">Euclidean Norm</span>
            </div>

            <div className="p-3 rounded-lg border border-zinc-800 bg-zinc-950 font-mono">
              <span className="text-zinc-400 text-xs block uppercase">Max Magnitude</span>
              <span className="text-zinc-200 font-bold text-lg">
                {stats.max_change_magnitude != null ? stats.max_change_magnitude.toFixed(3) : "N/A"}
              </span>
              <span className="text-xs text-zinc-500 block mt-0.5">Peak Anomaly</span>
            </div>
          </div>

          {changeData.interpretation && (
            <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950 font-mono text-xs text-zinc-300">
              <span className="font-bold text-amber-300 uppercase block mb-1">
                Automated Analytical Interpretation:
              </span>
              <p className="leading-relaxed">{changeData.interpretation}</p>
            </div>
          )}

          {changeData.disclaimer && (
            <div className="rounded-lg bg-amber-950/20 border border-amber-800/40 p-3 text-xs font-mono text-amber-300/80 flex items-start gap-2.5">
              <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <span>{changeData.disclaimer}</span>
            </div>
          )}
        </div>
      ) : (
        <div className="p-12 rounded-xl border border-dashed border-zinc-800 bg-zinc-950/40 text-center font-mono text-sm text-zinc-400 flex flex-col items-center gap-3">
          <History className="w-8 h-8 text-zinc-600" />
          <span>Select two compatible scenes to compare.</span>
        </div>
      )}
    </div>
  );
}
