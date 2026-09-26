"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";

/**
 * BootSplash — one-time app-wide logo pixelate-in splash.
 *
 * On a fresh page load, shows a full-viewport overlay with the BHARAT-SR
 * wordmark rendered via canvas, starting fully pixelated and automatically
 * resolving to crisp over ~1s, holding briefly, then fading out to reveal
 * the app underneath. Uses sessionStorage so it only plays once per browser
 * session (but re-plays on full refresh).
 *
 * Respects prefers-reduced-motion: skips animation entirely.
 */

interface BootSplashProps {
  children: React.ReactNode;
}

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

export function BootSplash({ children }: BootSplashProps) {
  const [showSplash, setShowSplash] = useState(false);
  const [fadeOut, setFadeOut] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const reducedMotionRef = useRef(false);

  // Determine on mount whether to show splash
  useEffect(() => {
    if (typeof window === "undefined") return;

    // Check reduced motion preference
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    reducedMotionRef.current = mq.matches;

    // Check sessionStorage gate
    const alreadyShown = sessionStorage.getItem("bsr_boot_splash_shown");
    if (alreadyShown) {
      setShowSplash(false);
      return;
    }

    // Show splash
    setShowSplash(true);
    sessionStorage.setItem("bsr_boot_splash_shown", "1");
  }, []);

  // Render the logo text to an offscreen canvas once
  const prepareLogoCanvas = useCallback(() => {
    const offscreen = document.createElement("canvas");
    const W = 640;
    const H = 240;
    offscreen.width = W;
    offscreen.height = H;
    const ctx = offscreen.getContext("2d");
    if (!ctx) return null;

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

    // Draw subtitle
    ctx.font = "400 13px 'JetBrains Mono', monospace";
    ctx.fillStyle = "#8a97a3";
    ctx.fillText("Physics-Constrained Super-Resolution", W / 2, H / 2 + 40);

    offscreenRef.current = offscreen;
    return offscreen;
  }, []);

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

      // Max block size → 1 (resolved)
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

    const offscreen = prepareLogoCanvas();
    if (!offscreen) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

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

    // Phase 1: pixelate → resolve over ~1s
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

    // Small delay before starting animation so fonts load
    const initTimer = setTimeout(() => {
      rafRef.current = requestAnimationFrame(animate);
    }, 100);

    return () => {
      clearTimeout(initTimer);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [showSplash, prepareLogoCanvas, renderAtProgress]);

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
            width={640}
            height={240}
            style={{
              maxWidth: "85vw",
              maxHeight: "30vh",
              width: "auto",
              height: "auto",
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
