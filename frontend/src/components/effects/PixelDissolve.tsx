"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { cn } from "@/lib/utils";

interface PixelDissolveProps {
  src: string;
  alt: string;
  className?: string;
  pixelSize?: number;
  duration?: number;
  trigger?: "hover" | "auto-loop" | "in-view";
  label?: string;
}

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

export function PixelDissolve({
  src,
  alt,
  className,
  pixelSize = 24,
  duration = 700,
  trigger = "hover",
  label = "Hover to resolve",
}: PixelDissolveProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const offscreenImgRef = useRef<HTMLImageElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const animStartRef = useRef<number | null>(null);
  const directionRef = useRef<"resolve" | "pixelate">("pixelate");
  const progressRef = useRef(0); // 0 = fully pixelated, 1 = fully resolved
  const isResolvedRef = useRef(false);

  const [isLoaded, setIsLoaded] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [canHover, setCanHover] = useState(true);
  const [displayGsd, setDisplayGsd] = useState(10.0);
  const [labelOpacity, setLabelOpacity] = useState(1);
  const [glowIntensity, setGlowIntensity] = useState(0);

  // Effective trigger: on touch-only devices, downgrade "hover" to "in-view"
  const effectiveTrigger = trigger === "hover" && !canHover ? "in-view" : trigger;

  // Check prefers-reduced-motion
  useEffect(() => {
    if (typeof window === "undefined") return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(mq.matches);
    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  // Detect pointer capability
  useEffect(() => {
    if (typeof window === "undefined") return;
    const mq = window.matchMedia("(hover: hover) and (pointer: fine)");
    setCanHover(mq.matches);
    const handler = (e: MediaQueryListEvent) => setCanHover(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  // Preload source image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      offscreenImgRef.current = img;
      setIsLoaded(true);
    };
    img.onerror = () => {
      // Still mark loaded to prevent infinite loading state
      setIsLoaded(true);
    };
    img.src = src;
  }, [src]);

  // Render a single frame at a given progress (0=pixelated, 1=resolved)
  const renderAtProgress = useCallback(
    (progress: number) => {
      const canvas = canvasRef.current;
      const img = offscreenImgRef.current;
      if (!canvas || !img) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;

      // Clamp progress
      const p = Math.max(0, Math.min(1, progress));

      // Interpolate block size: pixelSize → 1
      const currentBlock = Math.max(1, Math.round(pixelSize * (1 - p) + 1 * p));

      if (currentBlock <= 1) {
        // Fully resolved — draw crisp
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(img, 0, 0, w, h);
      } else {
        // Pixelate: downsample then upsample with no smoothing
        ctx.imageSmoothingEnabled = false;
        const coarseW = Math.max(1, Math.floor(w / currentBlock));
        const coarseH = Math.max(1, Math.floor(h / currentBlock));

        // Draw downsampled
        ctx.drawImage(img, 0, 0, coarseW, coarseH);
        // Upscale with nearest-neighbor
        ctx.drawImage(canvas, 0, 0, coarseW, coarseH, 0, 0, w, h);
      }

      // Update GSD readout
      const gsd = 10.0 - p * 7.5; // 10m → 2.5m
      setDisplayGsd(Math.round(gsd * 10) / 10);

      // Update label opacity (fade as we resolve)
      setLabelOpacity(Math.max(0, 1 - p * 1.5));

      // Update glow intensity
      setGlowIntensity(p);
    },
    [pixelSize]
  );

  // Initial render once image loads
  useEffect(() => {
    if (isLoaded && offscreenImgRef.current) {
      if (reducedMotion) {
        // Snap to resolved
        progressRef.current = 1;
        renderAtProgress(1);
      } else {
        progressRef.current = 0;
        renderAtProgress(0);
      }
    }
  }, [isLoaded, reducedMotion, renderAtProgress]);

  // Animation loop
  const startAnimation = useCallback(
    (direction: "resolve" | "pixelate") => {
      if (reducedMotion) {
        progressRef.current = direction === "resolve" ? 1 : 0;
        renderAtProgress(progressRef.current);
        return;
      }

      directionRef.current = direction;
      animStartRef.current = null;

      if (rafRef.current) cancelAnimationFrame(rafRef.current);

      const startProgress = progressRef.current;

      const animate = (timestamp: number) => {
        if (animStartRef.current === null) animStartRef.current = timestamp;
        const elapsed = timestamp - animStartRef.current;
        const rawT = Math.min(1, elapsed / duration);
        const easedT = easeOutCubic(rawT);

        if (direction === "resolve") {
          progressRef.current = startProgress + (1 - startProgress) * easedT;
        } else {
          progressRef.current = startProgress - startProgress * easedT;
        }

        renderAtProgress(progressRef.current);

        if (rawT < 1) {
          rafRef.current = requestAnimationFrame(animate);
        } else {
          isResolvedRef.current = direction === "resolve";
        }
      };

      rafRef.current = requestAnimationFrame(animate);
    },
    [duration, reducedMotion, renderAtProgress]
  );

  // Hover trigger
  const handlePointerEnter = useCallback(() => {
    if (effectiveTrigger !== "hover") return;
    startAnimation("resolve");
  }, [effectiveTrigger, startAnimation]);

  const handlePointerLeave = useCallback(() => {
    if (effectiveTrigger !== "hover") return;
    startAnimation("pixelate");
  }, [effectiveTrigger, startAnimation]);

  // In-view trigger
  useEffect(() => {
    if (effectiveTrigger !== "in-view" || !containerRef.current || !isLoaded) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const entry = entries[0];
        if (entry?.isIntersecting && entry.intersectionRatio >= 0.6) {
          startAnimation("resolve");
          observer.disconnect();
        }
      },
      { threshold: 0.6 }
    );
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [effectiveTrigger, isLoaded, startAnimation]);

  // Auto-loop trigger
  useEffect(() => {
    if (effectiveTrigger !== "auto-loop" || !isLoaded || reducedMotion) return;

    let cancelled = false;
    const RESOLVE_HOLD = 2000;
    const PIXELATE_HOLD = 2000;

    const cycle = () => {
      if (cancelled) return;
      startAnimation("resolve");
      setTimeout(() => {
        if (cancelled) return;
        setTimeout(() => {
          if (cancelled) return;
          startAnimation("pixelate");
          setTimeout(() => {
            if (cancelled) return;
            setTimeout(cycle, PIXELATE_HOLD);
          }, duration);
        }, RESOLVE_HOLD);
      }, duration);
    };

    // Start first cycle after a short delay
    const initTimer = setTimeout(cycle, 1000);

    return () => {
      cancelled = true;
      clearTimeout(initTimer);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [effectiveTrigger, isLoaded, reducedMotion, startAnimation, duration]);

  // Border glow color — interpolate from amber to cyan as resolve progresses
  const glowColor = `rgba(${Math.round(245 - glowIntensity * 205)}, ${Math.round(
    158 + glowIntensity * 66
  )}, ${Math.round(11 + glowIntensity * 187)}, ${0.15 + glowIntensity * 0.45})`;

  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <div
        ref={containerRef}
        onPointerEnter={handlePointerEnter}
        onPointerLeave={handlePointerLeave}
        className="relative overflow-hidden rounded-xl select-none"
        style={{
          aspectRatio: "1 / 1",
          boxShadow: `0 0 ${12 + glowIntensity * 24}px ${glowColor}, inset 0 0 1px ${glowColor}`,
          border: `1px solid ${glowColor}`,
          transition: "box-shadow 0.3s ease, border-color 0.3s ease",
        }}
      >
        <canvas
          ref={canvasRef}
          width={512}
          height={512}
          className="w-full h-full object-cover block bg-zinc-950"
          role="img"
          aria-label={alt}
        />

        {/* Small bottom-left caption — fades out as resolved */}
        {label && labelOpacity > 0.01 && (
          <div
            className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-zinc-950/80 font-mono text-[10px] text-zinc-400 pointer-events-none select-none"
            style={{ opacity: labelOpacity, transition: "opacity 0.15s ease" }}
          >
            {label}
          </div>
        )}
      </div>

      {/* GSD readout — outside the image, beneath it */}
      <div className="flex items-center justify-between px-1 font-mono text-[11px]">
        <span className="text-zinc-500">
          GSD:{" "}
          <span
            className="font-bold"
            style={{
              color:
                displayGsd <= 3
                  ? "var(--bsr-phosphor, #28e0c6)"
                  : displayGsd <= 6
                  ? "var(--bsr-signal, #ffb454)"
                  : "var(--bsr-ink-dim, #8a97a3)",
            }}
          >
            {displayGsd.toFixed(1)}m
          </span>
        </span>
        <span className="text-zinc-600 text-[10px]">
          {displayGsd <= 3 ? "Resolved" : displayGsd >= 9 ? "Raw sensor" : "Resolving…"}
        </span>
      </div>
    </div>
  );
}
