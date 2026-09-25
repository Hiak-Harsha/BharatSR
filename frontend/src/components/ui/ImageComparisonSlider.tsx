"use client";

import React, { useState, useRef, useCallback } from "react";
import { MultiSpectralViews } from "@/lib/api-client";
import { BandViewSwitcher } from "./BandViewSwitcher";
import { BandViewMode } from "@/lib/store";
import { cn } from "@/lib/utils";
import { Maximize2, Columns, SplitSquareVertical, Sparkles, Crosshair, Scan } from "lucide-react";
import { PixelResolveCanvas } from "@/components/effects/PixelResolveCanvas";

export interface ComparisonLayer {
  id: string;
  label: string;
  tag?: string;
  image: string;
  views?: MultiSpectralViews;
  isGroundTruth?: boolean;
}

export interface ImageComparisonSliderProps {
  // Composable layers format
  layers?: ComparisonLayer[];
  initialLeftId?: string;
  initialRightId?: string;

  // Legacy compatibility props
  beforeSrc?: string;
  afterSrc?: string;
  bicubicSrc?: string;
  groundTruthSrc?: string;
  uncertaintySrc?: string;
  errorMapSrc?: string;
  beforeViews?: MultiSpectralViews;
  afterViews?: MultiSpectralViews;
  bicubicViews?: MultiSpectralViews;
  groundTruthViews?: MultiSpectralViews;
  beforeLabel?: string;
  afterLabel?: string;

  // Shared props
  onInspectPixel?: (x: number, y: number) => void;
  inspectedPoint?: { x: number; y: number } | null;
  imageDimensions?: { width: number; height: number };
  className?: string;
}

export function ImageComparisonSlider({
  layers: inputLayers,
  initialLeftId,
  initialRightId,
  beforeSrc,
  afterSrc,
  bicubicSrc,
  groundTruthSrc,
  uncertaintySrc,
  errorMapSrc,
  beforeViews,
  afterViews,
  bicubicViews,
  groundTruthViews,
  beforeLabel = "LR Input (10m)",
  afterLabel = "BharatSR Output (2.5m-equiv)",
  onInspectPixel,
  inspectedPoint,
  imageDimensions,
  className,
}: ImageComparisonSliderProps) {
  // Construct layers from input or legacy props
  const layers: ComparisonLayer[] = inputLayers || [
    ...(beforeSrc ? [{ id: "lr", label: beforeLabel, tag: "10m Raw", image: beforeSrc, views: beforeViews }] : []),
    ...(bicubicSrc ? [{ id: "bicubic", label: "Bicubic Interpolation", tag: "Baseline", image: bicubicSrc, views: bicubicViews }] : []),
    ...(afterSrc ? [{ id: "sr", label: afterLabel, tag: "BharatSR", image: afterSrc, views: afterViews }] : []),
    ...(groundTruthSrc ? [{ id: "gt", label: "Reference Target", tag: "Demonstration Reference", image: groundTruthSrc, views: groundTruthViews, isGroundTruth: true }] : []),
    ...(uncertaintySrc ? [{ id: "uncertainty", label: "Predicted Uncertainty", tag: "Predicted σ", image: uncertaintySrc }] : []),
    ...(errorMapSrc ? [{ id: "error", label: "Absolute Deviation |SR - Reference|", tag: "L1 Deviation", image: errorMapSrc }] : []),
  ];

  const defaultLeft = initialLeftId || layers[0]?.id || "lr";
  const defaultRight = initialRightId || layers[layers.length > 2 ? 2 : 1]?.id || layers[1]?.id || "sr";

  const [leftLayerId, setLeftLayerId] = useState<string>(defaultLeft);
  const [rightLayerId, setRightLayerId] = useState<string>(defaultRight);
  const [sliderPos, setSliderPos] = useState<number>(50);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [layoutMode, setLayoutMode] = useState<"slider" | "side-by-side" | "quad" | "lens">("slider");
  const [activeBand, setActiveBand] = useState<BandViewMode>("composite");

  const containerRef = useRef<HTMLDivElement>(null);

  const leftLayer = layers.find((l) => l.id === leftLayerId) || layers[0];
  const rightLayer = layers.find((l) => l.id === rightLayerId) || layers[1] || layers[0];

  const getImageForBand = (layer?: ComparisonLayer): string => {
    if (!layer) return "";
    if (layer.views && layer.views[activeBand]) {
      return layer.views[activeBand]!;
    }
    return layer.image;
  };

  const handlePointerDown = () => setIsDragging(true);
  const handlePointerUp = () => setIsDragging(false);

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!isDragging || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const clientX = e.clientX;
      const pos = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
      setSliderPos(pos);
    },
    [isDragging]
  );

  const handleInspect = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!onInspectPixel) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const relY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    const targetW = imageDimensions?.width || 256;
    const targetH = imageDimensions?.height || 256;
    const px = Math.floor(relX * targetW);
    const py = Math.floor(relY * targetH);
    onInspectPixel(px, py);
  };

  return (
    <div className={cn("flex flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-950/80 p-4", className)}>
      {/* Top Controls: Layers & Mode */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-800 pb-3">
        {/* Layer Selectors */}
        <div className="flex flex-wrap items-center gap-2 text-xs font-mono">
          <div className="flex items-center gap-1.5 bg-zinc-900 border border-zinc-800 px-2 py-1 rounded-md">
            <span className="text-zinc-500 font-bold uppercase">Left:</span>
            <select
              value={leftLayerId}
              onChange={(e) => setLeftLayerId(e.target.value)}
              aria-label="Left Comparison Layer"
              className="bg-transparent text-amber-300 font-semibold focus:outline-none cursor-pointer"
            >
              {layers.map((l) => (
                <option key={l.id} value={l.id} className="bg-zinc-900 text-zinc-200">
                  {l.label} ({l.tag || l.id})
                </option>
              ))}
            </select>
          </div>

          <span className="text-zinc-600 font-bold">VS</span>

          <div className="flex items-center gap-1.5 bg-zinc-900 border border-zinc-800 px-2 py-1 rounded-md">
            <span className="text-zinc-500 font-bold uppercase">Right:</span>
            <select
              value={rightLayerId}
              onChange={(e) => setRightLayerId(e.target.value)}
              aria-label="Right Comparison Layer"
              className="bg-transparent text-cyan-300 font-semibold focus:outline-none cursor-pointer"
            >
              {layers.map((l) => (
                <option key={l.id} value={l.id} className="bg-zinc-900 text-zinc-200">
                  {l.label} ({l.tag || l.id})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Band Switcher & Mode buttons */}
        <div className="flex flex-wrap items-center gap-2">
          <BandViewSwitcher activeBand={activeBand} onBandChange={setActiveBand} />

          <div className="inline-flex rounded-lg border border-zinc-800 bg-zinc-900 p-0.5 text-zinc-400">
            <button
              type="button"
              onClick={() => setLayoutMode("slider")}
              title="Split Slider View"
              className={cn("p-1.5 rounded hover:text-zinc-100", layoutMode === "slider" && "bg-zinc-800 text-amber-400")}
            >
              <SplitSquareVertical className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={() => setLayoutMode("side-by-side")}
              title="Side-by-Side View"
              className={cn("p-1.5 rounded hover:text-zinc-100", layoutMode === "side-by-side" && "bg-zinc-800 text-amber-400")}
            >
              <Columns className="w-4 h-4" />
            </button>
            <button
              type="button"
              onClick={() => setLayoutMode("lens")}
              title="Interactive Optical Lens (Resolving Pixelation 10m -> 2.5m)"
              className={cn(
                "p-1.5 rounded hover:text-zinc-100 flex items-center gap-1 px-2 text-[11px] font-mono transition",
                layoutMode === "lens" && "bg-amber-500/20 text-amber-400 border border-amber-500/40 font-bold"
              )}
            >
              <Crosshair className="w-3.5 h-3.5" />
              <span>Optical Lens</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Viewport */}
      {layoutMode === "lens" ? (
        <div className="relative aspect-square w-full max-h-[580px] overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 shadow-2xl">
          <PixelResolveCanvas
            src={getImageForBand(rightLayer) || rightLayer?.image || afterSrc || beforeSrc || ""}
            alt={rightLayer?.label || "Optical Resolving Lens"}
            className="w-full h-full"
            overlayLabel="Hover / drag cursor to resolve 10m raw sensor pixels to 2.5m analytical clarity"
          />
        </div>
      ) : layoutMode === "slider" ? (
        <div
          ref={containerRef}
          onPointerDown={handlePointerDown}
          onPointerUp={handlePointerUp}
          onPointerMove={handlePointerMove}
          onClick={handleInspect}
          className="relative aspect-square w-full max-h-[580px] overflow-hidden rounded-lg border border-zinc-800 bg-black select-none cursor-crosshair shadow-2xl"
        >
          {/* Base Layer (Right) */}
          {rightLayer && (
            <img
              src={getImageForBand(rightLayer)}
              alt={rightLayer.label}
              className="absolute inset-0 w-full h-full object-contain pointer-events-none"
            />
          )}

          {/* Top Layer (Left, clipped by slider) */}
          {leftLayer && (
            <div
              className="absolute inset-0 overflow-hidden pointer-events-none"
              style={{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }}
            >
              <img
                src={getImageForBand(leftLayer)}
                alt={leftLayer.label}
                className="absolute inset-0 w-full h-full object-contain"
              />
            </div>
          )}

          {/* Slider Divider Bar */}
          <div
            className="absolute top-0 bottom-0 w-0.5 bg-amber-400 shadow-[0_0_12px_rgba(251,191,36,0.8)] cursor-ew-resize z-20 pointer-events-auto"
            style={{ left: `${sliderPos}%` }}
          >
            <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-zinc-900 border-2 border-amber-400 flex items-center justify-center text-amber-400 shadow-lg text-[10px] font-mono font-bold">
              ↔
            </div>
          </div>

          {/* Pixel Inspection Marker */}
          {inspectedPoint && (
            <div
              className="absolute w-4 h-4 -translate-x-1/2 -translate-y-1/2 border-2 border-cyan-400 rounded-full bg-cyan-400/20 pointer-events-none z-30 animate-ping"
              style={{
                left: `${(inspectedPoint.x / 256) * 100}%`,
                top: `${(inspectedPoint.y / 256) * 100}%`,
              }}
            />
          )}

          {/* Persistent Labels */}
          <div className="absolute bottom-3 left-3 bg-zinc-950/80 backdrop-blur-md px-2.5 py-1 rounded border border-zinc-800 text-[11px] font-mono text-amber-300 pointer-events-none z-10">
            {leftLayer?.label}
          </div>
          <div className="absolute bottom-3 right-3 bg-zinc-950/80 backdrop-blur-md px-2.5 py-1 rounded border border-zinc-800 text-[11px] font-mono text-cyan-300 pointer-events-none z-10">
            {rightLayer?.label}
          </div>
        </div>
      ) : (
        /* Side by Side Mode */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div
            onClick={handleInspect}
            className="relative aspect-square w-full rounded-lg border border-zinc-800 bg-black overflow-hidden cursor-crosshair"
          >
            {leftLayer && (
              <img
                src={getImageForBand(leftLayer)}
                alt={leftLayer.label}
                className="w-full h-full object-contain pointer-events-none"
              />
            )}
            <div className="absolute bottom-2 left-2 bg-zinc-950/80 px-2 py-0.5 rounded text-[11px] font-mono text-amber-300">
              {leftLayer?.label}
            </div>
          </div>

          <div
            onClick={handleInspect}
            className="relative aspect-square w-full rounded-lg border border-zinc-800 bg-black overflow-hidden cursor-crosshair"
          >
            {rightLayer && (
              <img
                src={getImageForBand(rightLayer)}
                alt={rightLayer.label}
                className="w-full h-full object-contain pointer-events-none"
              />
            )}
            <div className="absolute bottom-2 right-2 bg-zinc-950/80 px-2 py-0.5 rounded text-[11px] font-mono text-cyan-300">
              {rightLayer?.label}
            </div>
          </div>
        </div>
      )}

      {/* Footer Instructions */}
      <div className="flex items-center justify-between text-[11px] font-mono text-zinc-500 pt-1">
        <span>Click on canvas to inspect spectral signature profile at (x, y)</span>
        {inspectedPoint && (
          <span className="text-cyan-400">
            Selected: [{inspectedPoint.x}, {inspectedPoint.y}]
          </span>
        )}
      </div>
    </div>
  );
}

export default ImageComparisonSlider;
