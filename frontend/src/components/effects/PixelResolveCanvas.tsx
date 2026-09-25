"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Sparkles, Scan, Crosshair, Eye, RefreshCw, ZoomIn } from "lucide-react";

interface PixelResolveCanvasProps {
  src: string;
  alt?: string;
  className?: string;
  minBlock?: number;
  maxBlock?: number;
  radius?: number;
  overlayLabel?: string;
  showControls?: boolean;
}

export function PixelResolveCanvas({
  src,
  alt = "Satellite Pixel-Resolving Lens",
  className,
  minBlock = 1,
  maxBlock = 18,
  radius = 110,
  overlayLabel = "Hover / Drag cursor to resolve 10m raw pixels into 2.5m analytical clarity",
  showControls = true,
}: PixelResolveCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const offscreenImgRef = useRef<HTMLImageElement | null>(null);
  const rafIdRef = useRef<number | null>(null);
  const autoAnimRef = useRef<number | null>(null);

  const [isHovered, setIsHovered] = useState(false);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);
  const [autoScanAngle, setAutoScanAngle] = useState(0);
  const [autoScanEnabled, setAutoScanEnabled] = useState(true);
  const [currentRadius, setCurrentRadius] = useState(radius);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [isInView, setIsInView] = useState(true);
  const [isLoaded, setIsLoaded] = useState(false);
  const [lensMode, setLensMode] = useState<"interactive" | "raw" | "resolved">("interactive");

  // Check prefers-reduced-motion media query
  useEffect(() => {
    if (typeof window === "undefined") return;
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(mediaQuery.matches);
    if (mediaQuery.matches) {
      setAutoScanEnabled(false);
    }

    const handler = (e: MediaQueryListEvent) => {
      setReducedMotion(e.matches);
      if (e.matches) setAutoScanEnabled(false);
    };
    mediaQuery.addEventListener("change", handler);
    return () => mediaQuery.removeEventListener("change", handler);
  }, []);

  // IntersectionObserver to pause rendering when off-screen
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        setIsInView(entry?.isIntersecting ?? true);
      },
      { threshold: 0.1 }
    );
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Preload and retain source image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      offscreenImgRef.current = img;
      setIsLoaded(true);
    };
    img.src = src;
  }, [src]);

  // Main rendering engine
  const renderFrame = useCallback(
    (focalX?: number, focalY?: number) => {
      const canvas = canvasRef.current;
      const img = offscreenImgRef.current;
      if (!canvas || !img || !isInView) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;

      // Mode: Forced Full Resolved
      if (lensMode === "resolved") {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(img, 0, 0, width, height);
        return;
      }

      // Mode: Forced Raw 10m Pixel Blocks
      if (lensMode === "raw") {
        ctx.imageSmoothingEnabled = false;
        const coarseW = Math.max(1, Math.floor(width / maxBlock));
        const coarseH = Math.max(1, Math.floor(height / maxBlock));
        ctx.drawImage(img, 0, 0, coarseW, coarseH);
        ctx.drawImage(canvas, 0, 0, coarseW, coarseH, 0, 0, width, height);

        // Grid lines overlay to emphasize raw 10m pixels
        ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
        ctx.lineWidth = 1;
        for (let x = 0; x < width; x += maxBlock) {
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, height);
          ctx.stroke();
        }
        for (let y = 0; y < height; y += maxBlock) {
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(width, y);
          ctx.stroke();
        }
        return;
      }

      // Interactive Lens Mode:
      // Determine active focal point (cursor if hovered, else auto-scan position or center)
      let cx = focalX;
      let cy = focalY;

      if (cx === undefined || cy === undefined) {
        if (cursorPos) {
          cx = cursorPos.x;
          cy = cursorPos.y;
        } else if (autoScanEnabled && !reducedMotion) {
          // Smooth orbital path across the center
          cx = width * 0.5 + Math.cos(autoScanAngle) * (width * 0.28);
          cy = height * 0.5 + Math.sin(autoScanAngle * 1.5) * (height * 0.22);
        } else {
          // Center default lens
          cx = width * 0.5;
          cy = height * 0.5;
        }
      }

      // STEP 1: Draw coarse 10m pixelated background across the whole tile
      ctx.imageSmoothingEnabled = false;
      const coarseW = Math.max(1, Math.floor(width / maxBlock));
      const coarseH = Math.max(1, Math.floor(height / maxBlock));
      ctx.drawImage(img, 0, 0, coarseW, coarseH);
      ctx.drawImage(canvas, 0, 0, coarseW, coarseH, 0, 0, width, height);

      // Faint sensor grid pattern on coarse areas
      ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
      ctx.lineWidth = 1;
      const step = maxBlock;
      for (let x = 0; x < width; x += step * 2) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += step * 2) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // STEP 2: Circular Lens Sharpening
      // Save canvas state and create circular clip around focal point
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, currentRadius, 0, Math.PI * 2);
      ctx.clip();

      // Inside clip: draw full-resolution crisp 2.5m image!
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(img, 0, 0, width, height);

      // Subtle high-resolution sub-pixel mesh within resolved lens
      ctx.strokeStyle = "rgba(6, 182, 212, 0.12)";
      ctx.lineWidth = 0.5;
      for (let x = Math.floor(cx - currentRadius); x <= cx + currentRadius; x += 12) {
        ctx.beginPath();
        ctx.moveTo(x, cy - currentRadius);
        ctx.lineTo(x, cy + currentRadius);
        ctx.stroke();
      }
      ctx.restore();

      // STEP 3: Space Technology HUD & Lens Boundary
      ctx.save();

      // Glowing lens boundary ring
      ctx.beginPath();
      ctx.arc(cx, cy, currentRadius, 0, Math.PI * 2);
      ctx.strokeStyle = isHovered ? "rgba(245, 158, 11, 0.95)" : "rgba(6, 182, 212, 0.85)";
      ctx.lineWidth = 2.5;
      ctx.shadowColor = isHovered ? "rgba(245, 158, 11, 0.6)" : "rgba(6, 182, 212, 0.5)";
      ctx.shadowBlur = 10;
      ctx.stroke();

      // Outer dashed tracking ring
      ctx.shadowBlur = 0;
      ctx.beginPath();
      ctx.arc(cx, cy, currentRadius + 7, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
      ctx.lineWidth = 1;
      ctx.setLineDash([6, 6]);
      ctx.stroke();

      // Center crosshair
      ctx.setLineDash([]);
      ctx.strokeStyle = isHovered ? "rgba(245, 158, 11, 0.9)" : "rgba(6, 182, 212, 0.9)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx - 10, cy);
      ctx.lineTo(cx + 10, cy);
      ctx.moveTo(cx, cy - 10);
      ctx.lineTo(cx, cy + 10);
      ctx.stroke();

      // Small center dot
      ctx.beginPath();
      ctx.arc(cx, cy, 2, 0, Math.PI * 2);
      ctx.fillStyle = isHovered ? "#ffb454" : "#38bdf8";
      ctx.fill();

      // Target Coordinates & GSD HUD Badge
      const badgeW = 124;
      const badgeH = 22;
      const badgeX = Math.min(width - badgeW - 10, Math.max(10, cx - badgeW / 2));
      const badgeY = cy + currentRadius + 14 < height - 35 ? cy + currentRadius + 14 : cy - currentRadius - 32;

      ctx.fillStyle = "rgba(9, 12, 16, 0.9)";
      ctx.fillRect(badgeX, badgeY, badgeW, badgeH);
      ctx.strokeStyle = isHovered ? "rgba(245, 158, 11, 0.8)" : "rgba(6, 182, 212, 0.8)";
      ctx.lineWidth = 1;
      ctx.strokeRect(badgeX, badgeY, badgeW, badgeH);

      ctx.fillStyle = isHovered ? "#ffb454" : "#38bdf8";
      ctx.font = "bold 9px monospace";
      ctx.fillText(isHovered ? "LENS: 2.5m RESOLVED" : "AUTO-SCAN: 2.5m", badgeX + 8, badgeY + 14);

      // Outside indicator badge
      ctx.fillStyle = "rgba(9, 12, 16, 0.85)";
      ctx.fillRect(10, 10, 136, 20);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
      ctx.strokeRect(10, 10, 136, 20);
      ctx.fillStyle = "#a1a1aa";
      ctx.font = "9px monospace";
      ctx.fillText("OUTSIDE: 10m RAW GSD", 16, 23);

      ctx.restore();
    },
    [autoScanAngle, autoScanEnabled, cursorPos, currentRadius, isHovered, isInView, lensMode, maxBlock, reducedMotion]
  );

  // Auto-scan animation loop when idle
  useEffect(() => {
    if (!isLoaded || !isInView || reducedMotion || !autoScanEnabled || isHovered) {
      if (autoAnimRef.current) cancelAnimationFrame(autoAnimRef.current);
      return;
    }

    let lastTime = performance.now();
    const animate = (time: number) => {
      const delta = (time - lastTime) / 1000;
      lastTime = time;
      setAutoScanAngle((prev) => (prev + delta * 0.8) % (Math.PI * 2));
      autoAnimRef.current = requestAnimationFrame(animate);
    };

    autoAnimRef.current = requestAnimationFrame(animate);
    return () => {
      if (autoAnimRef.current) cancelAnimationFrame(autoAnimRef.current);
    };
  }, [isLoaded, isInView, reducedMotion, autoScanEnabled, isHovered]);

  // Re-render when angle or hover state changes
  useEffect(() => {
    if (isLoaded) {
      renderFrame();
    }
  }, [isLoaded, renderFrame]);

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * (canvasRef.current?.width || rect.width);
    const y = ((e.clientY - rect.top) / rect.height) * (canvasRef.current?.height || rect.height);

    setCursorPos({ x, y });
    setIsHovered(true);

    if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    rafIdRef.current = requestAnimationFrame(() => renderFrame(x, y));
  };

  const handlePointerEnter = (e: React.PointerEvent<HTMLDivElement>) => {
    setIsHovered(true);
    handlePointerMove(e);
  };

  const handlePointerLeave = () => {
    setIsHovered(false);
    setCursorPos(null);
    if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    rafIdRef.current = requestAnimationFrame(() => renderFrame());
  };

  return (
    <div className={cn("flex flex-col gap-2.5", className)}>
      <div
        ref={containerRef}
        onPointerMove={handlePointerMove}
        onPointerEnter={handlePointerEnter}
        onPointerLeave={handlePointerLeave}
        className="relative group overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 select-none cursor-crosshair shadow-2xl transition-all"
        style={{ aspectRatio: "1 / 1" }}
      >
        <canvas
          ref={canvasRef}
          width={512}
          height={512}
          className="w-full h-full object-cover block"
        />

        {/* Top Right Status Badge */}
        <div className="absolute top-2.5 right-2.5 flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-zinc-950/85 border border-zinc-800 backdrop-blur-md shadow-md pointer-events-none">
          <Scan className="w-3.5 h-3.5 text-amber-400 animate-pulse" />
          <span className="font-mono text-[10px] uppercase font-bold text-amber-300">
            {isHovered ? "Manual Optical Focus" : "Optical Sensor Simulator"}
          </span>
        </div>

        {/* Bottom Overlay Prompt */}
        <div className="absolute bottom-2.5 inset-x-2.5 flex items-center justify-between px-3 py-1.5 rounded-lg bg-zinc-950/90 border border-zinc-800 backdrop-blur-md pointer-events-none">
          <span className="font-mono text-[10px] text-zinc-300 truncate flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
            {isHovered ? "Focusing 2.5m resolution under cursor" : overlayLabel}
          </span>
          <span className="hidden sm:inline-flex items-center gap-1 font-mono text-[10px] text-amber-400 font-semibold shrink-0">
            <Crosshair className="w-3 h-3" />
            <span>{isHovered ? "ACTIVE" : "MOVE CURSOR"}</span>
          </span>
        </div>
      </div>

      {/* Interactive Toolbar Controls */}
      {showControls && (
        <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 rounded-lg bg-zinc-950 border border-zinc-800 font-mono text-xs">
          <div className="flex items-center gap-1">
            <span className="text-[10px] text-zinc-500 uppercase mr-1">Lens Mode:</span>
            <button
              type="button"
              onClick={() => setLensMode("interactive")}
              className={cn(
                "px-2 py-0.5 rounded text-[11px] font-semibold transition",
                lensMode === "interactive"
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                  : "text-zinc-400 hover:text-zinc-200"
              )}
            >
              Resolving Lens
            </button>
            <button
              type="button"
              onClick={() => setLensMode("raw")}
              className={cn(
                "px-2 py-0.5 rounded text-[11px] font-semibold transition",
                lensMode === "raw"
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                  : "text-zinc-400 hover:text-zinc-200"
              )}
            >
              10m Raw
            </button>
            <button
              type="button"
              onClick={() => setLensMode("resolved")}
              className={cn(
                "px-2 py-0.5 rounded text-[11px] font-semibold transition",
                lensMode === "resolved"
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                  : "text-zinc-400 hover:text-zinc-200"
              )}
            >
              2.5m SR
            </button>
          </div>

          <div className="flex items-center gap-3">
            <label className="flex items-center gap-1.5 text-[11px] text-zinc-400 cursor-pointer">
              <input
                type="checkbox"
                checked={autoScanEnabled}
                onChange={(e) => setAutoScanEnabled(e.target.checked)}
                className="rounded border-zinc-700 bg-zinc-900 text-amber-400 focus:ring-0 w-3.5 h-3.5"
              />
              <span>Auto Sweep</span>
            </label>

            <div className="flex items-center gap-1 text-[11px] text-zinc-400">
              <ZoomIn className="w-3 h-3 text-cyan-400" />
              <span>Lens:</span>
              <button
                type="button"
                onClick={() => setCurrentRadius(80)}
                className={cn("px-1.5 py-0.2 rounded text-[10px]", currentRadius === 80 ? "text-amber-400 font-bold" : "text-zinc-500")}
              >
                80px
              </button>
              <button
                type="button"
                onClick={() => setCurrentRadius(120)}
                className={cn("px-1.5 py-0.2 rounded text-[10px]", currentRadius === 120 ? "text-amber-400 font-bold" : "text-zinc-500")}
              >
                120px
              </button>
              <button
                type="button"
                onClick={() => setCurrentRadius(160)}
                className={cn("px-1.5 py-0.2 rounded text-[10px]", currentRadius === 160 ? "text-amber-400 font-bold" : "text-zinc-500")}
              >
                160px
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
