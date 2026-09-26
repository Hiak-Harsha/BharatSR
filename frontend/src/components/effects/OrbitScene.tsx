"use client";

import React, { useEffect, useState } from "react";

/**
 * OrbitScene — renders the earth/orbit/satellite/beam DOM structure
 * that the existing CSS classes in globals.css expect (.bsr-earth, .bsr-orbit,
 * .bsr-sat, .bsr-beam, @keyframes bsr-spin).
 *
 * Respects prefers-reduced-motion: freezes the orbit animation but keeps
 * the scene visible at a static angle.
 */
export function OrbitScene({ className }: { className?: string }) {
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(mq.matches);
    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  return (
    <div
      className={className}
      style={{ position: "relative", width: "100%", height: "100%", overflow: "hidden" }}
      aria-hidden="true"
    >
      {/* Earth globe */}
      <div className="bsr-earth" />

      {/* Orbit ring with satellite */}
      <div
        className="bsr-orbit"
        style={reducedMotion ? { animationPlayState: "paused" } : undefined}
      >
        <div className="bsr-sat">
          {/* Inline satellite SVG — body + two solar panels */}
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* Solar panel left */}
            <rect x="0" y="5" width="5" height="6" rx="0.5" fill="var(--bsr-signal, #ffb454)" opacity="1.0" />
            <line x1="1" y1="8" x2="4" y2="8" stroke="rgba(0,0,0,0.3)" strokeWidth="0.5" />
            <line x1="2.5" y1="5.5" x2="2.5" y2="10.5" stroke="rgba(0,0,0,0.2)" strokeWidth="0.5" />
            {/* Body */}
            <rect x="5.5" y="4" width="5" height="8" rx="1" fill="#c4cdd8" />
            <circle cx="8" cy="7" r="1.2" fill="var(--bsr-phosphor, #28e0c6)" opacity="1.0" />
            <rect x="6.5" y="10" width="3" height="1.5" rx="0.3" fill="#8a97a3" />
            {/* Solar panel right */}
            <rect x="11" y="5" width="5" height="6" rx="0.5" fill="var(--bsr-signal, #ffb454)" opacity="1.0" />
            <line x1="12" y1="8" x2="15" y2="8" stroke="rgba(0,0,0,0.3)" strokeWidth="0.5" />
            <line x1="13.5" y1="5.5" x2="13.5" y2="10.5" stroke="rgba(0,0,0,0.2)" strokeWidth="0.5" />
          </svg>
        </div>
      </div>

      {/* Scan beam from satellite */}
      <div className="bsr-beam" />

      {/* Pulsing scan ring on earth */}
      <div className="bsr-scan-ring" />
    </div>
  );
}
