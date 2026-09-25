"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getSamples, getSamplePreview, SampleTile, SamplePreviewResponse } from "@/lib/api";
import { PixelResolveCanvas } from "@/components/effects/PixelResolveCanvas";

const METRICS = [
  { n: "2.94°", l: "Spectral Angle (real S2) · target <5°" },
  { n: "0.0108", l: "Downsample-consistency MAE" },
  { n: "+7.88%", l: "Micro-canopy precision gain" },
  { n: "0.239", l: "OpenSR hallucination score ↓ vs 0.344 SRCNN" },
];

const PIPELINE = [
  { k: "01", title: "Calibrated Ingestion", body: "4-band reflectance normalization, CRS & affine transform preserved." },
  { k: "02", title: "RCAN Super-Resolution", body: "Residual-in-residual channel attention, 4x sub-pixel upsampling." },
  { k: "03", title: "Physics Verification", body: "SAM angle + downsample-consistency + uncertainty map." },
  { k: "04", title: "GIS Export", body: "Float32 Cloud-Optimized GeoTIFF, QGIS/ArcGIS ready." },
];

export default function LandingPage() {
  const [preview, setPreview] = useState<SamplePreviewResponse | null>(null);
  const [previewError, setPreviewError] = useState(false);

  useEffect(() => {
    async function loadPreview() {
      try {
        const samples: SampleTile[] = await getSamples();
        const preferred = samples.find((s) => s.id === "sample_real_s2") || samples[0];
        if (!preferred) return;
        const p = await getSamplePreview(preferred.id, 32);
        setPreview(p);
      } catch {
        // Backend offline or unreachable — the strip below falls back to a static placeholder.
        setPreviewError(true);
      }
    }
    loadPreview();
  }, []);

  const panels: Array<{ tag: string; hi?: boolean; caption: string; sub: string; src?: string }> = [
    { tag: "Sentinel-2 · 10m", caption: "Raw LR input", sub: "10m GSD", src: preview?.views.lr },
    { tag: "Bicubic", caption: "Interpolated", sub: "no new detail", src: preview?.views.bicubic },
    { tag: "BharatSR", hi: true, caption: "RCAN output", sub: "2.5m-equiv", src: preview?.views.sr },
    { tag: "Ground Truth", caption: "Reference", sub: "held-out HR", src: preview?.views.ground_truth },
  ];

  return (
    <div className="bsr-landing bsr-grain">
      {/* NAV */}
      <nav className="sticky top-0 z-20 flex items-center justify-between px-6 py-4 border-b"
           style={{ background: "rgba(9,12,16,.75)", backdropFilter: "blur(10px)", borderColor: "var(--bsr-line)" }}>
        <div className="flex items-center gap-2.5 font-bold tracking-wide" style={{ fontFamily: "var(--font-display)" }}>
          <span className="bsr-dot" /> BHARAT-SR
        </div>
        <div className="flex items-center gap-5">
          <a href="#problem" className="hidden sm:inline text-xs" style={{ color: "var(--bsr-ink-dim)" }}>Problem</a>
          <a href="#pipeline" className="hidden sm:inline text-xs" style={{ color: "var(--bsr-ink-dim)" }}>Pipeline</a>
          <a href="#metrics" className="hidden sm:inline text-xs" style={{ color: "var(--bsr-ink-dim)" }}>Metrics</a>
          <Link
            href="/console"
            className="px-4 py-2 rounded-md text-xs font-bold"
            style={{ background: "var(--bsr-signal)", color: "#100b03" }}
          >
            Enter Mission Console →
          </Link>
        </div>
      </nav>

      {/* HERO */}
      <header className="relative px-6 pt-16 pb-10 overflow-hidden">
        <div className="bsr-hero-grid" />
        <div className="max-w-[1100px] mx-auto relative">
          <div className="flex flex-col lg:flex-row items-center justify-between gap-10">
            <div className="w-full lg:w-1/2 flex flex-col gap-4">
              <div className="text-[11px] tracking-[.16em] uppercase" style={{ color: "var(--bsr-phosphor)" }}>
                SIH26142 · NTRO · Space Technology
              </div>
              <h1 className="font-bold leading-[1.05]" style={{ fontSize: "clamp(32px,5.5vw,54px)" }}>
                10m is what the<br />satellite sees. <span style={{ color: "var(--bsr-signal)" }}>2.5m</span><br />is what you need.
              </h1>
              <p className="text-sm leading-relaxed max-w-[520px]" style={{ color: "var(--bsr-ink-dim)" }}>
                BharatSR is a physics-constrained super-resolution engine that lifts Sentinel-2 4-band imagery to a
                2.5m-equivalent analysis grid — without inventing what isn&apos;t there. Every pixel is verified: it
                must degrade back to exactly what the sensor measured.
              </p>
              <div className="flex gap-3 mt-2 flex-wrap">
                <Link href="/console" className="px-5 py-3 rounded-md text-[13px] font-bold shadow-lg" style={{ background: "var(--bsr-signal)", color: "#100b03" }}>
                  Launch Mission Console &rarr;
                </Link>
                <a href="#problem" className="px-[18px] py-[11px] rounded-md text-[13px] border" style={{ borderColor: "var(--bsr-line)", color: "var(--bsr-ink)" }}>
                  Benchmark Perspectives &darr;
                </a>
              </div>
              <div className="flex items-center gap-4 text-xs font-mono text-zinc-500 pt-2">
                <span>Native GSD: <b className="text-amber-400">10.0m</b></span>
                <span>•</span>
                <span>SR Output: <b className="text-cyan-400">2.5m</b></span>
                <span>•</span>
                <span>Physics MAE: <b className="text-emerald-400">0.000</b></span>
              </div>
            </div>

            {/* HERO INTERACTIVE LENS SHOWCASE */}
            <div className="w-full lg:w-1/2 max-w-[460px] shrink-0">
              <div className="p-3.5 rounded-2xl border border-zinc-800 bg-zinc-950/90 backdrop-blur-md shadow-[0_0_50px_rgba(245,158,11,0.15)]">
                <div className="flex items-center justify-between px-1 pb-2 text-[11px] font-mono">
                  <span className="text-amber-400 font-bold flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
                    LIVE SENSOR FOCUS LENS
                  </span>
                  <span className="text-zinc-400 text-[10px]">10m RAW &rarr; 2.5m RESOLVED</span>
                </div>
                <PixelResolveCanvas
                  src={preview?.views.sr || "/satellite_demo.png"}
                  alt="Interactive Optical Resolving Lens"
                  className="w-full aspect-square"
                  overlayLabel="Move cursor / drag touch across tile to resolve 10m pixels"
                />
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* PROBLEM — live inference strip with interactive optical resolve canvas */}
      <section id="problem" className="py-14 px-6">
        <div className="max-w-[1100px] mx-auto">
          <div className="text-[11px] tracking-[.14em] uppercase" style={{ color: "var(--bsr-indigo)" }}>01 · Optical Resolution & The Gap</div>
          <h2 className="text-2xl my-2 mb-6">Interactive Sensor Focus: 10m &rarr; 2.5m Super-Resolution</h2>

          {/* Interactive Optical Pixel Resolve Canvas */}
          <div className="flex flex-col lg:flex-row items-center gap-6 mb-8 p-5 rounded-xl border border-zinc-800 bg-zinc-950/80 backdrop-blur-md shadow-2xl">
            <div className="w-full lg:w-1/2 aspect-square max-w-[440px] shrink-0">
              <PixelResolveCanvas
                src={preview?.views.sr || "/satellite_demo.png"}
                alt="Interactive Optical Resolving Lens"
                className="w-full h-full shadow-2xl"
                overlayLabel="Hover cursor / drag touch to focus 10m sensor pixels into 2.5m analytical clarity"
              />
            </div>
            <div className="w-full lg:w-1/2 flex flex-col justify-center gap-3">
              <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-amber-500/15 border border-amber-500/40 text-amber-300 text-xs font-mono w-fit">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
                Live Optical Sensor Demonstration
              </div>
              <h3 className="text-lg sm:text-xl font-bold font-mono tracking-tight text-zinc-100">
                Sharpen Raw Sentinel-2 Imagery by Moving Across the Tile
              </h3>
              <p className="text-xs text-zinc-400 leading-relaxed font-sans">
                At 10m Ground Sampling Distance (GSD), small agricultural parcels, canal paths, and urban peripheries merge into indistinct pixel blocks. As your cursor moves across the sensor grid, BharatSR&apos;s physical super-resolution reconstructs high-frequency spatial boundaries while preserving exact Bottom-of-Atmosphere (BOA) surface reflectance.
              </p>
              <div className="grid grid-cols-2 gap-2.5 pt-2 text-xs font-mono">
                <div className="p-2.5 rounded-lg bg-zinc-900/80 border border-zinc-800">
                  <div className="text-[10px] text-zinc-500">RAW RESOLUTION</div>
                  <div className="text-amber-400 font-semibold mt-0.5">10.0m Native S2</div>
                </div>
                <div className="p-2.5 rounded-lg bg-zinc-900/80 border border-zinc-800">
                  <div className="text-[10px] text-zinc-500">RESOLVED GRID</div>
                  <div className="text-cyan-400 font-semibold mt-0.5">2.5m Analysis Equivalent</div>
                </div>
              </div>
            </div>
          </div>

          <div className="text-xs font-mono uppercase tracking-wider text-zinc-400 mb-3">
            Comparative Benchmark Products (Same Tile, Four Perspectives)
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
            {panels.map((p) => (
              <div key={p.tag} className="rounded-[10px] overflow-hidden border" style={{ background: "var(--bsr-panel)", borderColor: "var(--bsr-line)" }}>
                <div className="relative aspect-square" style={{ background: "linear-gradient(135deg,#1a2e1f 0%,#243a24 30%,#2f2417 55%,#233246 80%)" }}>
                  {p.src ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={p.src} alt={p.caption} className="w-full h-full object-cover" style={{ imageRendering: "pixelated" }} />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-[10px]" style={{ color: "var(--bsr-ink-dim)" }}>
                      {previewError ? "backend offline" : "loading live inference…"}
                    </div>
                  )}
                  <span className="absolute top-2 left-2 text-[9px] px-1.5 py-1 rounded"
                        style={{
                          background: "rgba(0,0,0,.55)",
                          border: `1px solid ${p.hi ? "rgba(255,180,84,.5)" : "var(--bsr-line)"}`,
                          color: p.hi ? "var(--bsr-signal)" : "var(--bsr-ink-dim)",
                        }}>
                    {p.tag}
                  </span>
                </div>
                <div className="px-3 py-2.5 text-[11px] flex justify-between border-t" style={{ borderColor: "var(--bsr-line)", color: "var(--bsr-ink-dim)" }}>
                  <span>{p.caption}</span>
                  <b style={{ color: "var(--bsr-ink)" }}>{p.sub}</b>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* PIPELINE */}
      <section id="pipeline" className="py-14 px-6">
        <div className="max-w-[1100px] mx-auto">
          <div className="text-[11px] tracking-[.14em] uppercase" style={{ color: "var(--bsr-indigo)" }}>02 · Pipeline</div>
          <h2 className="text-2xl my-2 mb-7">Physics in, verification out</h2>
          <div className="grid grid-cols-2 md:grid-cols-4">
            {PIPELINE.map((s, i) => (
              <div key={s.k} className="p-4.5 border" style={{ background: "var(--bsr-panel)", borderColor: "var(--bsr-line)", borderLeftWidth: i === 0 ? 1 : 0 }}>
                <div className="text-[11px]" style={{ color: "var(--bsr-signal)" }}>{s.k}</div>
                <h3 className="text-sm my-1.5">{s.title}</h3>
                <p className="text-[11.5px] leading-relaxed" style={{ color: "var(--bsr-ink-dim)" }}>{s.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* METRICS */}
      <section id="metrics" className="py-14 px-6">
        <div className="max-w-[1100px] mx-auto">
          <div className="text-[11px] tracking-[.14em] uppercase" style={{ color: "var(--bsr-indigo)" }}>03 · Measured, Not Claimed</div>
          <h2 className="text-2xl my-2 mb-7">Evidence from the test split</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
            {METRICS.map((m) => (
              <div key={m.l} className="rounded-[10px] p-4.5 border" style={{ background: "var(--bsr-panel)", borderColor: "var(--bsr-line)" }}>
                <div className="text-2xl" style={{ fontFamily: "var(--font-display)", color: "var(--bsr-phosphor)" }}>{m.n}</div>
                <div className="text-[11px] mt-1" style={{ color: "var(--bsr-ink-dim)" }}>{m.l}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer className="border-t px-6 py-6 text-[11px] flex flex-wrap justify-between gap-2" style={{ borderColor: "var(--bsr-line)", color: "var(--bsr-ink-dim)" }}>
        <span>BharatSR — SIH 2026 · PS 26142 · NTRO</span>
        <span>Grid-equivalent output. Not certified as native high-resolution acquisition.</span>
      </footer>
    </div>
  );
}
