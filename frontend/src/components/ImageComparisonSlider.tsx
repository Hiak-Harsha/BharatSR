"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { MultiSpectralViews } from "@/lib/api";

type ViewMode = "slider" | "side-by-side" | "evidence-4view" | "ground-truth" | "uncertainty" | "error-map";
type SpectralBand = "rgb" | "cir" | "ndvi" | "nir" | "red" | "green" | "blue" | "error";

interface Props {
  beforeSrc: string; // fallback LR
  afterSrc: string;  // fallback SR
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
  onInspectPixel?: (x: number, y: number) => void;
  inspectedPoint?: { x: number; y: number } | null;
  uncertaintyThreshold?: number;
}

export default function ImageComparisonSlider({
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
  uncertaintyThreshold = 0,
}: Props) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const [activeView, setActiveView] = useState<ViewMode>("slider");
  const [activeBand, setActiveBand] = useState<SpectralBand>("rgb");
  const containerRef = useRef<HTMLDivElement>(null);

  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!onInspectPixel) return;
    const target = e.currentTarget;
    const rect = target.getBoundingClientRect();
    const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const relY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    const px = Math.floor(relX * 256);
    const py = Math.floor(relY * 256);
    onInspectPixel(px, py);
  };

  // Select image source based on active spectral band
  const currentBefore = (beforeViews && beforeViews[activeBand]) || beforeSrc;
  const currentAfter = (afterViews && afterViews[activeBand]) || afterSrc;
  const currentBicubic = (bicubicViews && bicubicViews[activeBand]) || bicubicSrc || beforeSrc;
  const currentGT = (groundTruthViews && groundTruthViews[activeBand]) || groundTruthSrc;
  const effectiveErrorMap = errorMapSrc || (afterViews && afterViews.error);

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
        <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-lg bg-slate-900 border border-slate-800 text-xs">
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
              onClick={() => setActiveView("evidence-4view")}
              className={`px-3 py-1 rounded-md font-medium transition flex items-center gap-1 ${
                activeView === "evidence-4view"
                  ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <span>🔬</span> Evidence 4-View
            </button>
          )}
          {groundTruthSrc && (
            <button
              onClick={() => setActiveView("ground-truth")}
              className={`px-3 py-1 rounded-md font-medium transition ${
                activeView === "ground-truth"
                  ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              3-Way Ground Truth
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
          {effectiveErrorMap && (
            <button
              onClick={() => setActiveView("error-map")}
              className={`px-3 py-1 rounded-md font-medium transition ${
                activeView === "error-map"
                  ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Absolute Error Map
            </button>
          )}
        </div>

        <span className="text-xs text-slate-500 font-mono hidden sm:inline">
          4x SR output on a 2.5m-equivalent grid
        </span>
      </div>

      {/* Spectral Band Switcher */}
      {activeView !== "uncertainty" && activeView !== "error-map" && (
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
              {effectiveErrorMap && (
                <button
                  onClick={() => setActiveBand("error")}
                  className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition flex items-center gap-1.5 ${
                    activeBand === "error"
                      ? "bg-amber-500/20 text-amber-300 border border-amber-500/50 shadow-sm"
                      : "text-slate-400 hover:text-slate-200 hover:bg-slate-900"
                  }`}
                >
                  <span>📉</span> Error Map (|SR-HR|)
                </button>
              )}
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
          onClick={handleCanvasClick}
          className="comparison-container aspect-square max-h-[520px] mx-auto cursor-crosshair select-none relative"
        >
          {/* Under image (SR Output) */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={currentAfter}
            alt={afterLabel}
            className="comparison-image w-full h-full object-contain"
          />

          {/* Uncertainty Threshold Alert Mask */}
          {uncertaintyThreshold > 0 && uncertaintySrc && (
            <div
              className="absolute inset-0 pointer-events-none z-10 mix-blend-screen opacity-75"
              style={{
                filter: `contrast(200%) brightness(${100 + uncertaintyThreshold * 200}%)`,
              }}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={uncertaintySrc}
                alt="Uncertainty Alert Mask"
                className="w-full h-full object-contain"
              />
            </div>
          )}

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

          {/* Tactical Coordinate Reticle (Inspected Point) */}
          {inspectedPoint && (
            <div
              className="absolute pointer-events-none z-30 -translate-x-1/2 -translate-y-1/2 flex items-center justify-center"
              style={{
                left: `${(inspectedPoint.x / 256) * 100}%`,
                top: `${(inspectedPoint.y / 256) * 100}%`,
              }}
            >
              <span className="w-6 h-6 rounded-full border-2 border-cyan-400 animate-ping absolute opacity-75" />
              <span className="w-3 h-3 rounded-full bg-cyan-400/90 border border-white shadow-lg" />
              <span className="absolute -top-6 left-3 text-[9px] font-mono bg-black/90 px-1.5 py-0.5 rounded text-cyan-300 border border-cyan-500/50 whitespace-nowrap shadow-md">
                Point ({inspectedPoint.x}, {inspectedPoint.y})
              </span>
            </div>
          )}

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
              <span className="text-[10px] text-slate-500 font-mono">Original Sentinel-2 (10m)</span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={currentBefore}
              alt={beforeLabel}
              className="w-full aspect-square rounded-lg object-contain bg-slate-950"
              style={{ imageRendering: "pixelated" }}
            />
          </div>

          <div 
            className="glass-panel p-3 border-cyan-500/30 cursor-crosshair relative"
            onClick={handleCanvasClick}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-cyan-300">
                {afterLabel} ({activeBand.toUpperCase()})
              </span>
              <span className="text-[10px] text-cyan-500 font-mono">4x Enhanced (2.5m-equiv Grid)</span>
            </div>
            {/* Tactical Coordinate Reticle (Inspected Point) */}
            {inspectedPoint && (
              <div
                className="absolute pointer-events-none z-30 -translate-x-1/2 -translate-y-1/2 flex items-center justify-center"
                style={{
                  left: `${(inspectedPoint.x / 256) * 100}%`,
                  top: `${(inspectedPoint.y / 256) * 100}%`,
                }}
              >
                <span className="w-6 h-6 rounded-full border-2 border-cyan-400 animate-ping absolute opacity-75" />
                <span className="w-3 h-3 rounded-full bg-cyan-400/90 border border-white shadow-lg" />
              </div>
            )}
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

      {/* Evidence Mode: Synchronized 4-View (LR, Bicubic, RCAN, HR) */}
      {activeView === "evidence-4view" && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <div className="glass-panel p-2.5">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-semibold text-slate-300">1. Low-Resolution Input</span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">10m GSD</span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={currentBefore}
              alt="LR Input"
              className="w-full aspect-square rounded-lg object-contain bg-slate-950"
              style={{ imageRendering: "pixelated" }}
            />
            <div className="mt-1.5 text-[10px] text-slate-500 text-center">Unprocessed Sentinel-2 L2A</div>
          </div>

          <div className="glass-panel p-2.5 border-amber-500/20">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-semibold text-amber-300">2. Bicubic Baseline</span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-950/60 text-amber-400 font-mono">2.5m Grid</span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={currentBicubic}
              alt="Bicubic Baseline"
              className="w-full aspect-square rounded-lg object-contain bg-slate-950"
              style={{ imageRendering: "pixelated" }}
            />
            <div className="mt-1.5 text-[10px] text-slate-500 text-center">Classical Polynomial Interpolation</div>
          </div>

          <div className="glass-panel p-2.5 border-cyan-500/40">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-semibold text-cyan-300">3. BharatSR (RCAN)</span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-300 font-mono">2.5m Grid</span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={currentAfter}
              alt="BharatSR RCAN"
              className="w-full aspect-square rounded-lg object-contain bg-slate-950 shadow-md shadow-cyan-950/40"
              style={{ imageRendering: "pixelated" }}
            />
            <div className="mt-1.5 text-[10px] text-cyan-400/80 text-center">Physics-Constrained Channel Attention</div>
          </div>

          <div className="glass-panel p-2.5 border-emerald-500/40">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-semibold text-emerald-300">4. Reference Ground Truth</span>
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-300 font-mono">HR Sensor</span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={currentGT || currentAfter}
              alt="Ground Truth HR"
              className="w-full aspect-square rounded-lg object-contain bg-slate-950 shadow-md shadow-emerald-950/40"
              style={{ imageRendering: "pixelated" }}
            />
            <div className="mt-1.5 text-[10px] text-emerald-400/80 text-center">Optical Validation Target</div>
          </div>
        </div>
      )}

      {/* Ground Truth 3-Way View */}
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

      {/* Absolute Error Map View */}
      {activeView === "error-map" && effectiveErrorMap && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="glass-panel p-3 border-cyan-500/30">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-cyan-300">{afterLabel}</span>
              <span className="text-[10px] text-cyan-500 font-mono">Reconstructed Output</span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={afterSrc}
              alt="SR"
              className="w-full aspect-square rounded-lg object-contain bg-slate-950"
              style={{ imageRendering: "pixelated" }}
            />
          </div>

          <div className="glass-panel p-3 border-rose-500/40">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-rose-300">Absolute Error Map (|SR - HR|)</span>
              <span className="text-[10px] text-rose-400 font-mono">Plasma: Yellow/White = Higher Residual Error</span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={effectiveErrorMap}
              alt="Absolute Error Map"
              className="w-full aspect-square rounded-lg object-contain bg-slate-950 shadow-lg shadow-rose-950/40"
              style={{ imageRendering: "pixelated" }}
            />
          </div>
        </div>
      )}
    </div>
  );
}
