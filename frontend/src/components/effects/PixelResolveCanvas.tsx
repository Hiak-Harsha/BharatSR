"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { Sparkles, Scan, Crosshair } from "lucide-react";

interface PixelResolveCanvasProps {
  src: string;
  alt?: string;
  className?: string;
  minBlock?: number;
  maxBlock?: number;
  radius?: number;
  overlayLabel?: string;
}

export function PixelResolveCanvas({
  src,
  alt = "Satellite Pixel-Resolving Lens",
  className,
  minBlock = 1,
  maxBlock = 16,
  radius = 120,
  overlayLabel = "Interactive Optical Lens: Hover / Drag to Resolve 10m -> 2.5m",
}: PixelResolveCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const offscreenImgRef = useRef<HTMLImageElement | null>(null);
  const rafIdRef = useRef<number | null>(null);

  const [isHovered, setIsHovered] = useState(false);
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [isInView, setIsInView] = useState(true);
  const [isLoaded, setIsLoaded] = useState(false);

  // Check prefers-reduced-motion media query
  useEffect(() => {
    if (typeof window === "undefined") return;
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(mediaQuery.matches);

    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
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
      drawStaticOrResolved();
    };
    img.src = src;
  }, [src]);

  // Main rendering engine
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const img = offscreenImgRef.current;
    if (!canvas || !img || !isInView) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Reduced motion or no interaction: draw crisp or slightly coarse base
    if (reducedMotion || !cursorPos || !isHovered) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(img, 0, 0, width, height);

      // Subtle indicator badge
      ctx.fillStyle = "rgba(9, 9, 11, 0.75)";
      ctx.fillRect(8, height - 28, 175, 20);
      ctx.strokeStyle = "rgba(245, 158, 11, 0.4)";
      ctx.strokeRect(8, height - 28, 175, 20);

      ctx.fillStyle = "#f59e0b";
      ctx.font = "10px monospace";
      ctx.fillText("OPTICAL RESOLVER: IDLE", 14, height - 14);
      return;
    }

    const { x: cx, y: cy } = cursorPos;
    const step = maxBlock;

    // Draw coarse pixelated background first
    ctx.imageSmoothingEnabled = false;
    const coarseW = Math.max(1, Math.floor(width / maxBlock));
    const coarseH = Math.max(1, Math.floor(height / maxBlock));

    // Temporary downscale-upscale for background 10m pixel blocks
    ctx.drawImage(img, 0, 0, coarseW, coarseH);
    ctx.drawImage(canvas, 0, 0, coarseW, coarseH, 0, 0, width, height);

    // Block-by-block resolution sharpening inside cursor radius
    const startX = Math.max(0, Math.floor((cx - radius * 1.3) / step) * step);
    const endX = Math.min(width, Math.ceil((cx + radius * 1.3) / step) * step);
    const startY = Math.max(0, Math.floor((cy - radius * 1.3) / step) * step);
    const endY = Math.min(height, Math.ceil((cy + radius * 1.3) / step) * step);

    for (let bx = startX; bx < endX; bx += step) {
      for (let by = startY; by < endY; by += step) {
        const blockCenterX = bx + step / 2;
        const blockCenterY = by + step / 2;
        const dist = Math.hypot(blockCenterX - cx, blockCenterY - cy);

        if (dist > radius) continue;

        const factor = Math.min(1, dist / radius);
        // Nonlinear easing: sharpest at focal center
        const dynamicBlockSize = Math.max(
          minBlock,
          Math.round(minBlock + (maxBlock - minBlock) * Math.pow(factor, 1.8))
        );

        if (dynamicBlockSize <= 2) {
          // Full resolution sampling from clean source
          ctx.drawImage(img, bx, by, step, step, bx, by, step, step);
        } else {
          // Intermediate sub-resolution block
          const sampleW = Math.max(1, Math.floor(step / dynamicBlockSize));
          const sampleH = Math.max(1, Math.floor(step / dynamicBlockSize));

          // Draw sampled small tile and upscale back
          ctx.drawImage(
            img,
            bx, by, step, step,
            bx, by, sampleW, sampleH
          );
          ctx.drawImage(
            canvas,
            bx, by, sampleW, sampleH,
            bx, by, step, step
          );
        }
      }
    }

    // Space Technology Focal Reticle HUD
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, 32, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(6, 182, 212, 0.85)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.stroke();

    // Center crosshair
    ctx.setLineDash([]);
    ctx.strokeStyle = "rgba(245, 158, 11, 0.9)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy);
    ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8);
    ctx.lineTo(cx, cy + 8);
    ctx.stroke();

    // HUD Telemetry Badge
    ctx.fillStyle = "rgba(9, 9, 11, 0.85)";
    ctx.fillRect(cx + 14, cy - 24, 94, 18);
    ctx.strokeStyle = "rgba(6, 182, 212, 0.5)";
    ctx.strokeRect(cx + 14, cy - 24, 94, 18);

    ctx.fillStyle = "#38bdf8";
    ctx.font = "9px monospace";
    ctx.fillText("2.5m RESOLVED", cx + 18, cy - 12);
    ctx.restore();
  }, [cursorPos, isHovered, reducedMotion, isInView, minBlock, maxBlock, radius]);

  const drawStaticOrResolved = useCallback(() => {
    const canvas = canvasRef.current;
    const img = offscreenImgRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  }, []);

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * (canvasRef.current?.width || rect.width);
    const y = ((e.clientY - rect.top) / rect.height) * (canvasRef.current?.height || rect.height);

    setCursorPos({ x, y });

    if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    rafIdRef.current = requestAnimationFrame(draw);
  };

  const handlePointerEnter = () => {
    setIsHovered(true);
    if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    rafIdRef.current = requestAnimationFrame(draw);
  };

  const handlePointerLeave = () => {
    setIsHovered(false);
    setCursorPos(null);
    if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    rafIdRef.current = requestAnimationFrame(drawStaticOrResolved);
  };

  useEffect(() => {
    if (isLoaded) {
      draw();
    }
  }, [isLoaded, draw]);

  return (
    <div
      ref={containerRef}
      onPointerMove={handlePointerMove}
      onPointerEnter={handlePointerEnter}
      onPointerLeave={handlePointerLeave}
      className={cn(
        "relative group overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 select-none cursor-crosshair",
        className
      )}
    >
      <canvas
        ref={canvasRef}
        width={512}
        height={512}
        className="w-full h-full object-cover block transition-opacity duration-300"
      />

      {/* Top Banner Tag */}
      <div className="absolute top-2.5 left-2.5 flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-zinc-950/85 border border-zinc-800 backdrop-blur-md shadow-md">
        <Scan className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
        <span className="font-mono text-[10px] uppercase font-semibold text-zinc-300">
          Sensor Simulation: 10m &rarr; 2.5m
        </span>
      </div>

      {/* Bottom Hint Overlay */}
      <div className="absolute bottom-2.5 inset-x-2.5 flex items-center justify-between px-3 py-1.5 rounded-lg bg-zinc-950/85 border border-zinc-800/80 backdrop-blur-md">
        <span className="font-mono text-[10px] text-zinc-400 truncate">
          {overlayLabel}
        </span>
        <span className="hidden sm:inline-flex items-center gap-1 font-mono text-[10px] text-amber-400">
          <Crosshair className="w-3 h-3" />
          <span>Interactive</span>
        </span>
      </div>
    </div>
  );
}
