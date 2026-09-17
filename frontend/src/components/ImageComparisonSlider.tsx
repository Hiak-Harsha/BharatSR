"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { MultiSpectralViews } from "@/lib/api";

type ViewMode = "slider" | "side-by-side" | "ground-truth" | "uncertainty";
type SpectralBand = "rgb" | "cir" | "ndvi" | "nir" | "red" | "green" | "blue";

interface Props {
  beforeSrc: string; // fallback LR
  afterSrc: string;  // fallback SR
  groundTruthSrc?: string;
  uncertaintySrc?: string;
  beforeViews?: MultiSpectralViews;
  afterViews?: MultiSpectralViews;
  groundTruthViews?: MultiSpectralViews;
  beforeLabel?: string;
  afterLabel?: string;
}

export default function ImageComparisonSlider({
  beforeSrc,
  afterSrc,
  groundTruthSrc,
  uncertaintySrc,
  beforeViews,
  afterViews,
  groundTruthViews,
  beforeLabel = "LR Input (Upscaled)",
  afterLabel = "BharatSR Output (4x)",
}: Props) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const [activeView, setActiveView] = useState<ViewMode>("slider");
  const [activeBand, setActiveBand] = useState<SpectralBand>("rgb");
  const containerRef = useRef<HTMLDivElement>(null);

  // Select image source based on active spectral band
  const currentBefore = (beforeViews && beforeViews[activeBand]) || beforeSrc;
  const currentAfter = (afterViews && afterViews[activeBand]) || afterSrc;
  const currentGT = (groundTruthViews && groundTruthViews[activeBand]) || groundTruthSrc;

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = Math.max(0, Math.min((x / rect.width) * 100, 100));
    setSliderPosition(percent);
  }, []);

  const handleTouchMove = useCallback((e: TouchEvent) => {
    if (!isDragging) return;
    handleMove(e.touches[0].clientX);
  }, [isDragging, handleMove]);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return;
    handleMove(e.clientX);
  }, [isDragging, handleMove]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      window.addEventListener("touchmove", handleTouchMove);
      window.addEventListener("touchend", handleMouseUp);
    }
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleMouseUp);
    };
  }, [isDragging, handleMouseMove, handleMouseUp, handleTouchMove]);

  return (
    <div className="space-y-3">
      {/* View Mode Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 p-1 rounded-lg bg-slate-900 border border-slate-800 text-xs">
          <button
            onClick={() => setActiveView("slider")}
            className={`px-3 py-1 rounded-md font-medium transition ${
              activeView === "slider"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Split Slider
          </button>
          <button
            onClick={() => setActiveView("side-by-side")}
            className={`px-3 py-1 rounded-md font-medium transition ${
              activeView === "side-by-side"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Side-by-Side
          </button>
          {groundTruthSrc && (
            <button
              onClick={() => setActiveView("ground-truth")}
              className={`px-3 py-1 rounded-md font-medium transition ${
                activeView === "ground-truth"
                  ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Ground Truth
            </button>
          )}
          {uncertaintySrc && (
            <button
              onClick={() => setActiveView("uncertainty")}
              className={`px-3 py-1 rounded-md font-medium transition ${
                activeView === "uncertainty"
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Uncertainty Heatmap
            </button>
          )}
        </div>

        <span className="text-xs text-slate-500 font-mono hidden sm:inline">
          4x Physics-Preserving Spatial Resolution
        </span>
      </div>

      {/* Spectral Band Switcher */}
      {activeView !== "uncertainty" && (
        <div className="p-2 rounded-xl bg-slate-950/80 border border-slate-800/80 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider pl-1">
              Channel:
            </span>
            <div className="flex flex-wrap items-center gap-1 text-xs">
              <button
                onClick={() => setActiveBand("rgb")}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition flex items-center gap-1.5 ${
                  activeBand === "rgb"
                    ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                }`}
              >
                <span>🌈</span> True Color (RGB)
              </button>
              <button
                onClick={() => setActiveBand("cir")}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition flex items-center gap-1.5 ${
                  activeBand === "cir"
                    ? "bg-rose-500/20 text-rose-300 border border-rose-500/50 shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                }`}
              >
                <span>🔴</span> False Color CIR (NIR-R-G)
              </button>
              <button
                onClick={() => setActiveBand("ndvi")}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition flex items-center gap-1.5 ${
                  activeBand === "ndvi"
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 shadow-sm"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                }`}
              >
                <span>🌿</span> NDVI Vegetation Index
              </button>
              <div className="h-3.5 w-px bg-slate-800 mx-1 hidden md:block" />
              <button
                onClick={() => setActiveBand("nir")}
                className={`px-2 py-1 rounded-md text-[11px] font-mono transition ${
                  activeBand === "nir"
                    ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/50"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                }`}
              >
                NIR (B8)
              </button>
              <button
                onClick={() => setActiveBand("red")}
                className={`px-2 py-1 rounded-md text-[11px] font-mono transition ${
                  activeBand === "red"
                    ? "bg-rose-500/20 text-rose-300 border border-rose-500/50"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                }`}
              >
                Red (B4)
              </button>
              <button
                onClick={() => setActiveBand("green")}
                className={`px-2 py-1 rounded-md text-[11px] font-mono transition ${
                  activeBand === "green"
                    ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/50"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                }`}
              >
                Green (B3)
              </button>
              <button
                onClick={() => setActiveBand("blue")}
                className={`px-2 py-1 rounded-md text-[11px] font-mono transition ${
                  activeBand === "blue"
                    ? "bg-sky-500/20 text-sky-300 border border-sky-500/50"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                }`}
              >
                Blue (B2)
              </button>
            </div>
          </div>

          <span className="text-[10px] text-slate-500 font-mono hidden xl:inline">
            Active: {activeBand.toUpperCase()} Channel Analysis
          </span>
        </div>
      )}

      {/* Slider View */}
      {activeView === "slider" && (
        <div
          ref={containerRef}
          onMouseDown={() => setIsDragging(true)}
          onTouchStart={() => setIsDragging(true)}
          className="comparison-container aspect-square max-h-[520px] mx-auto cursor-ew-resize select-none relative"
        >
          {/* Under image (SR Output) */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={currentAfter}
            alt={afterLabel}
            className="comparison-image w-full h-full object-contain"
          />

          {/* Over image (LR Input clipped) */}
          <div
            className="absolute inset-0 overflow-hidden"
            style={{ width: `${sliderPosition}%` }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={currentBefore}
              alt={beforeLabel}
              style={{
                width: containerRef.current ? `${containerRef.current.clientWidth}px` : "100%",
                maxWidth: "none",
                height: "100%",
                objectFit: "contain",
              }}
              className="comparison-image"
            />
          </div>

          {/* Divider Handle */}
          <div
            className="slider-handle"
            style={{ left: `${sliderPosition}%` }}
          >
            <div className="slider-button">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M8 9l4-4 4 4m0 6l-4 4-4-4" />
              </svg>
            </div>
          </div>

          {/* Floating Badges */}
          <div className="absolute top-3 left-3 px-2.5 py-1 rounded bg-slate-950/80 backdrop-blur border border-slate-700/60 text-[11px] font-semibold text-slate-300 pointer-events-none">
            {beforeLabel} ({activeBand.toUpperCase()})
          </div>
          <div className="absolute top-3 right-3 px-2.5 py-1 rounded bg-cyan-950/80 backdrop-blur border border-cyan-700/60 text-[11px] font-semibold text-cyan-300 pointer-events-none">
            {afterLabel} ({activeBand.toUpperCase()})
          </div>
        </div>
      )}

      {/* Side-by-Side View */}
      {activeView === "side-by-side" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="glass-panel p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-slate-300">
                {beforeLabel} ({activeBand.toUpperCase()})
              </span>
              <span className="text-[10px] text-slate-500 font-mono">Original Sentinel-2/NAIP</span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={currentBefore}
              alt={beforeLabel}
              className="w-full aspect-square rounded-lg object-contain bg-slate-950"
              style={{ imageRendering: "pixelated" }}
            />
          </div>

          <div className="glass-panel p-3 border-cyan-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-cyan-300">
                {afterLabel} ({activeBand.toUpperCase()})
              </span>
              <span className="text-[10px] text-cyan-500 font-mono">4x Enhanced (BharatSR)</span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={currentAfter}
              alt={afterLabel}
              className="w-full aspect-square rounded-lg object-contain bg-slate-950 shadow-lg shadow-cyan-950/40"
              style={{ imageRendering: "pixelated" }}
            />
          </div>
        </div>
      )}

      {/* Ground Truth View */}
      {activeView === "ground-truth" && currentGT && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="glass-panel p-3">
            <div className="text-xs font-semibold text-slate-400 mb-2">
              1. LR Input ({activeBand.toUpperCase()})
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={currentBefore} alt="LR" className="w-full aspect-square rounded-lg object-contain bg-slate-950" style={{ imageRendering: "pixelated" }} />
          </div>
          <div className="glass-panel p-3 border-cyan-500/40">
            <div className="text-xs font-semibold text-cyan-300 mb-2">
              2. BharatSR Predicted ({activeBand.toUpperCase()})
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={currentAfter} alt="SR" className="w-full aspect-square rounded-lg object-contain bg-slate-950" style={{ imageRendering: "pixelated" }} />
          </div>
          <div className="glass-panel p-3 border-emerald-500/40">
            <div className="text-xs font-semibold text-emerald-300 mb-2">
              3. Reference Ground Truth ({activeBand.toUpperCase()})
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={currentGT} alt="HR" className="w-full aspect-square rounded-lg object-contain bg-slate-950" style={{ imageRendering: "pixelated" }} />
          </div>
        </div>
      )}

      {/* Uncertainty Heatmap View */}
      {activeView === "uncertainty" && uncertaintySrc && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="glass-panel p-3 border-cyan-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-cyan-300">{afterLabel}</span>
              <span className="text-[10px] text-cyan-500 font-mono">Predicted Reflectance (4-Band)</span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={afterSrc}
              alt="SR"
              className="w-full aspect-square rounded-lg object-contain bg-slate-950"
              style={{ imageRendering: "pixelated" }}
            />
          </div>

          <div className="glass-panel p-3 border-amber-500/40">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-amber-300">Spatial Uncertainty (Magma Heatmap)</span>
              <span className="text-[10px] text-amber-400 font-mono">Bright Yellow/White = Higher Variance</span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={uncertaintySrc}
              alt="Uncertainty Heatmap"
              className="w-full aspect-square rounded-lg object-contain bg-slate-950 shadow-lg shadow-amber-950/40"
              style={{ imageRendering: "pixelated" }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
