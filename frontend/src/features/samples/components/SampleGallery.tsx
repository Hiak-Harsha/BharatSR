"use client";

import React, { useRef, useEffect } from "react";
import { useSamples } from "../hooks/useSamples";
import { useConsoleStore } from "@/lib/store";
import { Layers, Upload, CheckCircle2, MapPin, Calendar, Globe2, Radio } from "lucide-react";
import { cn } from "@/lib/utils";

export function SampleGallery({ className }: { className?: string }) {
  const { data: samples, isLoading, isError } = useSamples();
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const setSelectedSample = useConsoleStore((s) => s.setSelectedSample);
  const customFile = useConsoleStore((s) => s.customFile);
  const setCustomFile = useConsoleStore((s) => s.setCustomFile);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-select sample_real_s2 on first load if available, or first available sample
  useEffect(() => {
    if (samples && samples.length > 0 && !selectedSample && !customFile) {
      const realSample = samples.find((s) => s.id === "sample_real_s2");
      setSelectedSample(realSample ? realSample.id : samples[0].id);
    }
  }, [samples, selectedSample, customFile, setSelectedSample]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setCustomFile(file);
    }
  };

  return (
    <div className={cn("flex flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-950 p-4", className)}>
      <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between border-b border-zinc-850 pb-2.5">
        <div className="flex items-center gap-2 min-w-0">
          <Layers className="w-4 h-4 text-cyan-400 shrink-0" />
          <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-zinc-200 truncate">
            Target Sentinel-2 Scene
          </h3>
        </div>
        <span className="font-mono text-xs text-zinc-400 font-medium shrink-0">
          {samples?.length || 0} Available Scenes
        </span>
      </div>

      {/* Upload Sentinel-2 GeoTIFF Option */}
      <div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".tif,.tiff"
          onChange={handleFileUpload}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className={cn(
            "w-full flex items-center justify-between p-3 rounded-lg border text-left transition font-mono text-xs",
            customFile
              ? "border-cyan-500/80 bg-cyan-950/20 text-cyan-200"
              : "border-dashed border-zinc-700 bg-zinc-900/40 text-zinc-300 hover:text-zinc-100 hover:border-zinc-500"
          )}
        >
          <div className="flex flex-col gap-0.5 truncate">
            <div className="flex items-center gap-2 truncate">
              <Upload className="w-4 h-4 shrink-0 text-cyan-400" />
              <span className="font-semibold truncate">
                {customFile ? `Uploaded: ${customFile.name}` : "Upload Sentinel-2 GeoTIFF"}
              </span>
            </div>
            <span className="text-[11px] text-zinc-400 pl-6">
              Required bands: B2 (Blue) · B3 (Green) · B4 (Red) · B8 (NIR)
            </span>
          </div>
          {customFile && <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />}
        </button>
      </div>

      {/* Pre-loaded Available Scenes Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
          <div className="h-32 bg-zinc-900 animate-pulse rounded-lg" />
          <div className="h-32 bg-zinc-900 animate-pulse rounded-lg" />
        </div>
      ) : isError ? (
        <div className="text-xs font-mono text-rose-400 p-3 bg-rose-950/20 border border-rose-800/40 rounded-lg">
          Failed to load available scenes list.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-2.5 max-h-[420px] overflow-y-auto pr-1">
          {samples?.map((sample) => {
            const isSelected = !customFile && selectedSample === sample.id;
            const refProvenance = sample.is_independent_hr
              ? "Independent HR Reference"
              : sample.has_ground_truth || (sample as any).has_reference
              ? "Demonstration-derived Reference"
              : "Unpaired Input";

            return (
              <button
                key={sample.id}
                type="button"
                onClick={() => setSelectedSample(sample.id)}
                className={cn(
                  "flex items-start gap-3 p-2.5 rounded-lg border text-left overflow-hidden transition relative group",
                  isSelected
                    ? "border-amber-500 bg-amber-950/20 shadow-[0_0_12px_rgba(245,158,11,0.2)]"
                    : "border-zinc-800 bg-zinc-900/60 hover:border-zinc-700"
                )}
              >
                {/* Thumbnail */}
                <div className="w-20 h-20 shrink-0 bg-black rounded border border-zinc-800 relative overflow-hidden">
                  <img
                    src={sample.views?.composite || sample.thumbnail}
                    alt={sample.title || sample.id}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  {isSelected && (
                    <div className="absolute top-1 right-1 p-0.5 rounded-full bg-amber-500 text-zinc-950">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                    </div>
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-1">
                    <span className="font-mono text-xs font-bold text-zinc-200 truncate">
                      {sample.title || sample.id}
                    </span>
                    <span className={cn(
                      "px-1.5 py-0.5 rounded text-[11px] font-mono shrink-0 font-medium",
                      sample.has_geo
                        ? "bg-emerald-950/60 text-emerald-300 border border-emerald-800/40"
                        : "bg-zinc-800/80 text-zinc-400"
                    )}>
                      {sample.has_geo ? (sample.crs || "Georeferenced") : "Demo Grid"}
                    </span>
                  </div>

                  <div className="flex items-center gap-1 font-mono text-[11px] text-zinc-400 truncate mt-1">
                    <Radio className="w-3 h-3 text-cyan-400 shrink-0" />
                    <span className="truncate">{sample.sensor || "Sentinel-2 MSI"}</span>
                  </div>

                  <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] font-mono text-zinc-400">
                    <span>GSD: {(sample as any).gsd || "10m"}</span>
                    <span>Grid: {sample.lr_size}</span>
                    <span>Bands: B2·B3·B4·B8</span>
                  </div>

                  <div className="mt-1 text-[11px] font-mono text-amber-400/90 truncate">
                    {refProvenance}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
