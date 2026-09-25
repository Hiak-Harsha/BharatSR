"use client";

import React from "react";
import { BandViewMode } from "@/lib/store";
import { cn } from "@/lib/utils";
import { LayoutGrid, Layers } from "lucide-react";

interface BandViewSwitcherProps {
  activeBand: BandViewMode;
  onBandChange: (band: BandViewMode) => void;
  availableBands?: BandViewMode[];
  className?: string;
}

const SUB_BANDS: { id: BandViewMode; label: string; desc: string }[] = [
  { id: "rgb", label: "RGB", desc: "True Color (B4, B3, B2)" },
  { id: "cir", label: "CIR", desc: "Color Infrared (B8, B4, B3)" },
  { id: "ndvi", label: "NDVI", desc: "Normalized Difference Veg. Index" },
  { id: "nir", label: "NIR", desc: "Near-Infrared (Band 8, 842nm)" },
  { id: "red", label: "RED", desc: "Red (Band 4, 665nm)" },
  { id: "green", label: "GRN", desc: "Green (Band 3, 560nm)" },
  { id: "blue", label: "BLU", desc: "Blue (Band 2, 490nm)" },
  { id: "error", label: "ERR", desc: "Error Map |SR - GT|" },
];

export function BandViewSwitcher({
  activeBand,
  onBandChange,
  availableBands,
  className,
}: BandViewSwitcherProps) {
  const isCompositeActive = activeBand === "composite";
  const subBands = availableBands
    ? SUB_BANDS.filter((b) => availableBands.includes(b.id))
    : SUB_BANDS;

  return (
    <div
      className={cn(
        "flex flex-col sm:flex-row sm:items-center gap-1.5 p-1.5 bg-zinc-950/90 border border-zinc-800 rounded-lg backdrop-blur-md shadow-lg",
        className
      )}
    >
      {/* Primary Analytical Product: All Bands Composite */}
      <button
        type="button"
        onClick={() => onBandChange("composite")}
        title="All Bands Together: True Color + CIR + NDVI + B2/B3/B4/B8 Analytical Master Grid"
        className={cn(
          "inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono font-semibold rounded-md transition-all shadow-sm",
          isCompositeActive
            ? "bg-amber-500/25 text-amber-300 border border-amber-500/80 shadow-[0_0_12px_rgba(245,158,11,0.25)]"
            : "text-zinc-300 bg-zinc-900/80 hover:text-zinc-100 hover:bg-zinc-800/80 border border-zinc-750"
        )}
      >
        <LayoutGrid className="w-3.5 h-3.5 text-amber-400 shrink-0" />
        <span>All Bands (Composite)</span>
        <span className="hidden sm:inline-block px-1 py-0.2 text-[9px] uppercase tracking-wider rounded bg-amber-500/20 text-amber-300 font-mono">
          Master
        </span>
      </button>

      {/* Sub-views Group */}
      <div className="flex items-center gap-1 flex-wrap pl-1 sm:border-l sm:border-zinc-800">
        <span className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider px-1 select-none">
          Sub-views:
        </span>
        {subBands.map((band) => {
          const isActive = activeBand === band.id;
          return (
            <button
              key={band.id}
              type="button"
              onClick={() => onBandChange(band.id)}
              title={band.desc}
              className={cn(
                "px-2 py-1 text-[11px] font-mono font-medium rounded transition-all",
                isActive
                  ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/60 shadow-[0_0_8px_rgba(6,182,212,0.2)]"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-850 border border-transparent"
              )}
            >
              {band.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

