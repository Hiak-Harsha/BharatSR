"use client";

import React from "react";
import { BandViewMode } from "@/lib/store";
import { cn } from "@/lib/utils";

interface BandViewSwitcherProps {
  activeBand: BandViewMode;
  onBandChange: (band: BandViewMode) => void;
  availableBands?: BandViewMode[];
  className?: string;
}

const ALL_BANDS: { id: BandViewMode; label: string; desc: string }[] = [
  { id: "rgb", label: "RGB", desc: "True Color (B4, B3, B2)" },
  { id: "cir", label: "CIR", desc: "Color Infrared (B8, B4, B3)" },
  { id: "ndvi", label: "NDVI", desc: "Normalized Difference Veg. Index" },
  { id: "nir", label: "NIR", desc: "Near-Infrared (Band 8)" },
  { id: "red", label: "RED", desc: "Red (Band 4)" },
  { id: "green", label: "GRN", desc: "Green (Band 3)" },
  { id: "blue", label: "BLU", desc: "Blue (Band 2)" },
  { id: "error", label: "ERR", desc: "Error Map |SR - GT|" },
];

export function BandViewSwitcher({
  activeBand,
  onBandChange,
  availableBands,
  className,
}: BandViewSwitcherProps) {
  const bands = availableBands
    ? ALL_BANDS.filter((b) => availableBands.includes(b.id))
    : ALL_BANDS;

  return (
    <div
      className={cn(
        "inline-flex flex-wrap items-center gap-1 p-1 bg-zinc-900/90 border border-zinc-800 rounded-lg backdrop-blur-sm",
        className
      )}
    >
      {bands.map((band) => {
        const isActive = activeBand === band.id;
        return (
          <button
            key={band.id}
            type="button"
            onClick={() => onBandChange(band.id)}
            title={band.desc}
            className={cn(
              "px-2.5 py-1 text-xs font-mono font-medium rounded transition-all",
              isActive
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/50 shadow-[0_0_10px_rgba(245,158,11,0.2)]"
                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60 border border-transparent"
            )}
          >
            {band.label}
          </button>
        );
      })}
    </div>
  );
}
