"use client";

import React, { useRef } from "react";
import { useSamples } from "../hooks/useSamples";
import { useConsoleStore } from "@/lib/store";
import { Layers, Upload, CheckCircle2, MapPin } from "lucide-react";
import { cn } from "@/lib/utils";

export function SampleGallery({ className }: { className?: string }) {
  const { data: samples, isLoading, isError } = useSamples();
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const setSelectedSample = useConsoleStore((s) => s.setSelectedSample);
  const customFile = useConsoleStore((s) => s.customFile);
  const setCustomFile = useConsoleStore((s) => s.setCustomFile);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setCustomFile(file);
    }
  };

  return (
    <div className={cn("flex flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-950 p-4", className)}>
      <div className="flex items-center justify-between border-b border-zinc-850 pb-2.5">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-cyan-400" />
          <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-zinc-200">
            Target Sentinel-2 Tile
          </h3>
        </div>
        <span className="font-mono text-[10px] text-zinc-500">
          {samples?.length || 0} Calibrated Tiles
        </span>
      </div>

      {/* Upload Custom Imagery Option */}
      <div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          onChange={handleFileUpload}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className={cn(
            "w-full flex items-center justify-between p-2.5 rounded-lg border text-left transition font-mono text-xs",
            customFile
              ? "border-cyan-500/80 bg-cyan-950/20 text-cyan-200"
              : "border-dashed border-zinc-700 bg-zinc-900/40 text-zinc-400 hover:text-zinc-200 hover:border-zinc-500"
          )}
        >
          <div className="flex items-center gap-2 truncate">
            <Upload className="w-4 h-4 shrink-0 text-cyan-400" />
            <span className="truncate">
              {customFile ? `Uploaded: ${customFile.name}` : "Upload Custom GeoTIFF / Image..."}
            </span>
          </div>
          {customFile && <CheckCircle2 className="w-4 h-4 text-cyan-400 shrink-0" />}
        </button>
      </div>

      {/* Pre-loaded Sample Tiles Grid */}
      {isLoading ? (
        <div className="grid grid-cols-2 gap-2">
          <div className="h-24 bg-zinc-900 animate-pulse rounded-lg" />
          <div className="h-24 bg-zinc-900 animate-pulse rounded-lg" />
        </div>
      ) : isError ? (
        <div className="text-xs font-mono text-rose-400 p-2.5 bg-rose-950/20 border border-rose-800/40 rounded-lg">
          Failed to load sample tiles list.
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2 max-h-[360px] overflow-y-auto pr-1">
          {samples?.map((sample) => {
            const isSelected = !customFile && selectedSample === sample.id;
            return (
              <button
                key={sample.id}
                type="button"
                onClick={() => setSelectedSample(sample.id)}
                className={cn(
                  "flex flex-col rounded-lg border text-left overflow-hidden transition relative group",
                  isSelected
                    ? "border-amber-500 bg-amber-950/20 shadow-[0_0_12px_rgba(245,158,11,0.2)]"
                    : "border-zinc-800 bg-zinc-900/60 hover:border-zinc-700"
                )}
              >
                {/* Thumbnail */}
                <div className="aspect-square w-full bg-black relative overflow-hidden">
                  <img
                    src={sample.views?.composite || sample.thumbnail}
                    alt={sample.title || sample.id}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  {isSelected && (
                    <div className="absolute top-1.5 right-1.5 p-1 rounded-full bg-amber-500 text-zinc-950">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                    </div>
                  )}
                  {sample.tactical_category && (
                    <span className="absolute bottom-1.5 left-1.5 px-1.5 py-0.5 rounded text-[9px] font-mono bg-zinc-950/80 text-amber-300 backdrop-blur-sm border border-zinc-800">
                      {sample.tactical_category}
                    </span>
                  )}
                </div>

                <div className="p-2">
                  <div className="font-mono text-xs font-bold text-zinc-200 truncate">
                    {sample.title || sample.id}
                  </div>
                  {sample.region && (
                    <div className="flex items-center gap-1 font-mono text-[10px] text-zinc-400 truncate mt-0.5">
                      <MapPin className="w-2.5 h-2.5 text-zinc-500 shrink-0" />
                      <span className="truncate">{sample.region}</span>
                    </div>
                  )}
                  <div className="mt-1 flex items-center justify-between text-[10px] font-mono text-zinc-500">
                    <span>{sample.lr_size}</span>
                    <span>{sample.has_ground_truth ? "GT Available" : "Unpaired"}</span>
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
