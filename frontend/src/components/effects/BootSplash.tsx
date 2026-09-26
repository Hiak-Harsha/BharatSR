"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";
import { usePathname } from "next/navigation";

/**
 * BootSplash — app-wide logo pixelate-in splash on page load/refresh.
 *
 * Plays on every full document load / hard refresh of any page in the app,
 * showing a full-viewport overlay with the BHARAT-SR wordmark and a route-aware
 * subtitle. The logo starts pixelated and resolves to crisp over ~1000ms using
 * device-pixel-ratio scaling, holds briefly (~500ms), and fades out (~400ms).
 * Total duration: under 2.5s.
 *
 * Because BootSplash is mounted in root layout.tsx, Next.js client-side
 * navigation (<Link>) does NOT re-mount the component, so it never replays
 * on in-app transitions. Hard refreshes replay it every time.
 *
 * Respects prefers-reduced-motion: skips animation entirely.
 */

interface BootSplashProps {
  children: React.ReactNode;
}

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

function getRouteSubtitle(pathname: string | null): string {
  if (!pathname || pathname === "/") return "Physics-Constrained Super-Resolution";
  if (pathname === "/console") return "Mission Console";
  if (pathname === "/console/run") return "Mission Console · Super-Resolution";
  if (pathname === "/console/compare") return "Mission Console · Model Benchmark";
  if (pathname === "/console/indices") return "Mission Console · Spectral Indices";
  if (pathname === "/console/crop-health") return "Mission Console · Crop Health";
  if (pathname === "/console/field-boundary") return "Mission Console · Field Boundaries";
  if (pathname === "/console/change-detection") return "Mission Console · Change Detection";
  if (pathname === "/console/downstream") return "Mission Console · Downstream Tasks";
  if (pathname === "/console/jobs") return "Mission Console · Batch Jobs";
  if (pathname === "/console/export") return "Mission Console · GeoTIFF & Reports";
  if (pathname === "/console/dataset") return "Mission Console · Dataset & Training";

  // Generic fallback from path segments
  const parts = pathname.split("/").filter(Boolean);
  if (parts.length > 0) {
    return parts
      .map((p) => p.charAt(0).toUpperCase() + p.slice(1).replace(/-/g, " "))
      .join(" · ");
  }
  return "Physics-Constrained Super-Resolution";
}

export function BootSplash({ children }: BootSplashProps) {
  const pathname = usePathname();
  const [showSplash, setShowSplash] = useState(false);
  const [fadeOut, setFadeOut] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const reducedMotionRef = useRef(false);

  // Trigger splash on initial document mount (hard load or refresh)
  useEffect(() => {
    if (typeof window === "undefined") return;

    // Check reduced motion preference
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    reducedMotionRef.current = mq.matches;

    // Show splash on every hard page load / browser refresh
    setShowSplash(true);
  }, []);

  // Render the logo text to an offscreen canvas with High-DPI support
  const prepareLogoCanvas = useCallback(
    (subtitle: string) => {
      const offscreen = document.createElement("canvas");
      const dpr = typeof window !== "undefined" ? Math.max(1, window.devicePixelRatio || 1) : 1;
      const baseW = 640;
      const baseH = 240;
      offscreen.width = baseW * dpr;
      offscreen.height = baseH * dpr;
      const ctx = offscreen.getContext("2d");
      if (!ctx) return null;

      ctx.scale(dpr, dpr);
      const W = baseW;
      const H = baseH;

      // Dark background matching --bsr-bg
      ctx.fillStyle = "#090c10";
      ctx.fillRect(0, 0, W, H);

      // Draw the phosphor dot
      const dotRadius = 8;
      const dotX = W / 2 - 145;
      const dotY = H / 2 + 2;
      ctx.beginPath();
      ctx.arc(dotX, dotY, dotRadius, 0, Math.PI * 2);
      ctx.fillStyle = "#28e0c6";
      ctx.fill();
      // Dot glow
      ctx.shadowColor = "#28e0c6";
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.arc(dotX, dotY, dotRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.shadowColor = "transparent";

      // Draw "BHARAT-SR" wordmark
      ctx.font = "bold 52px 'Space Grotesk', sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#e8edf2";
      ctx.fillText("BHARAT-SR", W / 2 + 10, H / 2);

      // Draw route-specific subtitle
      ctx.font = "400 13px 'JetBrains Mono', monospace";
      ctx.fillStyle = "#8a97a3";
      ctx.fillText(subtitle, W / 2, H / 2 + 40);

      offscreenRef.current = offscreen;
      return offscreen;
    },
    []
  );

  // Pixelation render at given progress (0 = fully pixelated, 1 = resolved)
  const renderAtProgress = useCallback(
    (progress: number) => {
      const canvas = canvasRef.current;
      const offscreen = offscreenRef.current;
      if (!canvas || !offscreen) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;
      const p = Math.max(0, Math.min(1, progress));

      // Continuous block calculation: 32 down to 1
      const maxBlock = 32;
      const currentBlock = Math.max(1, Math.round(maxBlock * (1 - p) + 1 * p));

      // Clear
      ctx.fillStyle = "#090c10";
      ctx.fillRect(0, 0, w, h);

      if (currentBlock <= 1) {
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(offscreen, 0, 0, w, h);
      } else {
        ctx.imageSmoothingEnabled = false;
        const coarseW = Math.max(1, Math.floor(w / currentBlock));
        const coarseH = Math.max(1, Math.floor(h / currentBlock));
        ctx.drawImage(offscreen, 0, 0, coarseW, coarseH);
        ctx.drawImage(canvas, 0, 0, coarseW, coarseH, 0, 0, w, h);
      }
    },
    []
  );

  // Run the animation sequence
  useEffect(() => {
    if (!showSplash) return;

    const subtitle = getRouteSubtitle(pathname);
    const offscreen = prepareLogoCanvas(subtitle);
    if (!offscreen) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = typeof window !== "undefined" ? Math.max(1, window.devicePixelRatio || 1) : 1;
    canvas.width = 640 * dpr;
    canvas.height = 240 * dpr;

    // If reduced motion, show resolved immediately then fade
    if (reducedMotionRef.current) {
      renderAtProgress(1);
      const fadeTimer = setTimeout(() => setFadeOut(true), 600);
      const hideTimer = setTimeout(() => setShowSplash(false), 1000);
      return () => {
        clearTimeout(fadeTimer);
        clearTimeout(hideTimer);
      };
    }

    // Start pixelated
    renderAtProgress(0);

    // Phase 1: pixelate → resolve over ~1000ms
    const RESOLVE_DURATION = 1000;
    const HOLD_DURATION = 500;
    const FADE_DURATION = 400;

    let startTime: number | null = null;

    const animate = (timestamp: number) => {
      if (startTime === null) startTime = timestamp;
      const elapsed = timestamp - startTime;
      const rawT = Math.min(1, elapsed / RESOLVE_DURATION);
      const easedT = easeOutCubic(rawT);

      renderAtProgress(easedT);

      if (rawT < 1) {
        rafRef.current = requestAnimationFrame(animate);
      } else {
        // Phase 2: hold, then fade
        setTimeout(() => {
          setFadeOut(true);
          setTimeout(() => {
            setShowSplash(false);
          }, FADE_DURATION);
        }, HOLD_DURATION);
      }
    };

    // Brief delay so font assets have a tick to render
    const initTimer = setTimeout(() => {
      rafRef.current = requestAnimationFrame(animate);
    }, 80);

    return () => {
      clearTimeout(initTimer);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [showSplash, pathname, prepareLogoCanvas, renderAtProgress]);

  return (
    <>
      {showSplash && (
        <div
          className="boot-splash-overlay"
          style={{
            position: "fixed",
            inset: 0,
            zIndex: 9999,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            backgroundColor: "#090c10",
            opacity: fadeOut ? 0 : 1,
            transition: fadeOut ? "opacity 400ms ease-out" : "none",
            pointerEvents: fadeOut ? "none" : "auto",
          }}
        >
          <canvas
            ref={canvasRef}
            style={{
              maxWidth: "85vw",
              maxHeight: "30vh",
              width: "640px",
              height: "240px",
              imageRendering: "auto",
            }}
            aria-label="BharatSR boot splash"
          />
        </div>
      )}
      {children}
    </>
  );
}
