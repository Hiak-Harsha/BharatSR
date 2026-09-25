"use client";

import React from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { useConsoleStore } from "@/lib/store";
import { fetchHealth, getJobs } from "@/lib/api-client";
import {
  Sparkles,
  GitCompare,
  BarChart3,
  Sprout,
  Maximize2,
  History,
  Target,
  ListFilter,
  Download,
  Database,
  ArrowRight,
  Layers,
  CheckCircle2,
  Clock,
  HardDrive,
  Cpu,
} from "lucide-react";
import { cn } from "@/lib/utils";

export default function ConsoleOverviewPage() {
  const selectedModel = useConsoleStore((s) => s.selectedModel);
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const currentRunId = useConsoleStore((s) => s.currentRunId);
  const currentSession = useConsoleStore((s) => s.currentSession);

  // Queries for live metrics
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
  });

  const { data: jobs } = useQuery({
    queryKey: ["jobs"],
    queryFn: getJobs,
  });

  const activeJobsCount = jobs?.filter((j) => j.status === "processing" || j.status === "pending").length ?? 0;
  const completedJobsCount = jobs?.filter((j) => j.status === "completed").length ?? 0;
  const exportsCount = 2;

  const WORKFLOW_STEPS = [
    {
      step: "01",
      title: "Resolve Scene",
      desc: "Select Sentinel-2 10m L2A multi-band reflectance and execute physics-constrained 4× super-resolution to 2.5m.",
      href: "/console/run",
      icon: Sparkles,
      color: "amber",
      badge: selectedModel.toUpperCase(),
    },
    {
      step: "02",
      title: "Analyze & Verify",
      desc: "Inspect spectral fidelity, NDVI/NDWI indices, field boundaries, crop health maps, and pixel-level uncertainty.",
      href: "/console/indices",
      icon: BarChart3,
      color: "cyan",
      badge: currentRunId ? "RUN READY" : "AWAITING RUN",
    },
    {
      step: "03",
      title: "Batch Operations",
      desc: "Queue entire satellite orbits, parallelize tile reconstruction across GPU workers, and monitor telemetry.",
      href: "/console/jobs",
      icon: ListFilter,
      color: "indigo",
      badge: `${activeJobsCount} ACTIVE`,
    },
    {
      step: "04",
      title: "Export GeoTIFF",
      desc: "Package georeferenced Float32 rasters with preserved CRS, affine transformations, and QA compliance reports.",
      href: "/console/export",
      icon: Download,
      color: "emerald",
      badge: `${exportsCount} FILES`,
    },
  ];

  const QUICK_WORKSPACES = [
    { name: "Model Comparison", desc: "Side-by-side 4-model ablation grid", href: "/console/compare", icon: GitCompare },
    { name: "Crop Health & Canopy", desc: "Vegetation vigor, LAI & stress classification", href: "/console/crop-health", icon: Sprout },
    { name: "Field Boundary Detection", desc: "Sub-pixel agricultural parcel delineation", href: "/console/field-boundary", icon: Maximize2 },
    { name: "Multi-Temporal Change", desc: "Bi-temporal difference & urban expansion mapping", href: "/console/change-detection", icon: History },
    { name: "Downstream Segmentation", desc: "Road, water, and building extraction masks", href: "/console/downstream", icon: Target },
    { name: "Dataset & Training", desc: "Data provenance, preprocessed pairs & training loss", href: "/console/dataset", icon: Database },
  ];

  return (
    <div className="w-full max-w-[1600px] mx-auto flex flex-col gap-8 pb-12">
      {/* Overview Hero Banner */}
      <div className="relative overflow-hidden rounded-3xl border border-zinc-800/80 bg-gradient-to-br from-zinc-950 via-zinc-900/60 to-zinc-950 p-6 sm:p-10 shadow-2xl backdrop-blur-md">
        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-amber-300 font-mono text-xs uppercase tracking-wider mb-4">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              NTRO SIH26142 Mission Control
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight">
              Satellite Super-Resolution <span className="text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-amber-200">Mission Console</span>
            </h1>
            <p className="mt-3 text-sm sm:text-base text-zinc-300 font-sans leading-relaxed">
              Transform medium-resolution Sentinel-2 multispectral imagery into 2.5m-equivalent high-fidelity surface reflectance products with strict spectral conservation and quantified uncertainty.
            </p>
          </div>

          <div className="flex flex-wrap sm:flex-nowrap gap-3 shrink-0">
            <Link
              href="/console/run"
              className="px-6 py-3 rounded-xl bg-amber-500 hover:bg-amber-400 text-zinc-950 font-bold text-sm transition shadow-lg shadow-amber-500/25 flex items-center gap-2"
            >
              <Sparkles className="w-4 h-4" />
              <span>Launch Super-Resolution</span>
              <ArrowRight className="w-4 h-4" />
            </Link>
            <Link
              href="/console/dataset"
              className="px-5 py-3 rounded-xl bg-zinc-900 hover:bg-zinc-800 text-zinc-200 border border-zinc-700 font-semibold text-sm transition flex items-center gap-2"
            >
              <Database className="w-4 h-4 text-zinc-400" />
              <span>Dataset &amp; Models</span>
            </Link>
          </div>
        </div>
      </div>

      {/* At-a-Glance Summary Telemetry Tiles */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl border border-zinc-800/80 bg-zinc-950/70 backdrop-blur-md flex items-center justify-between">
          <div>
            <div className="text-xs font-mono text-zinc-500 uppercase tracking-wider">Target Scene</div>
            <div className="text-base font-bold text-zinc-100 mt-1 truncate max-w-[160px]" title={selectedSample || "sample_real_s2"}>
              {selectedSample || "sample_real_s2"}
            </div>
            <div className="text-[11px] text-zinc-400 font-mono mt-0.5">Sentinel-2 10m L2A</div>
          </div>
          <div className="w-11 h-11 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-amber-400">
            <Layers className="w-5 h-5" />
          </div>
        </div>

        <div className="p-5 rounded-2xl border border-zinc-800/80 bg-zinc-950/70 backdrop-blur-md flex items-center justify-between">
          <div>
            <div className="text-xs font-mono text-zinc-500 uppercase tracking-wider">Active Pipeline</div>
            <div className="text-base font-bold text-amber-400 mt-1 uppercase">
              {selectedModel} · 4× SR
            </div>
            <div className="text-[11px] text-zinc-400 font-mono mt-0.5">
              Device: {health?.device || "CPU"}
            </div>
          </div>
          <div className="w-11 h-11 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
            <Cpu className="w-5 h-5" />
          </div>
        </div>

        <div className="p-5 rounded-2xl border border-zinc-800/80 bg-zinc-950/70 backdrop-blur-md flex items-center justify-between">
          <div>
            <div className="text-xs font-mono text-zinc-500 uppercase tracking-wider">Batch Queue</div>
            <div className="text-base font-bold text-zinc-100 mt-1">
              {activeJobsCount} Active <span className="text-xs font-normal text-zinc-500">({completedJobsCount} done)</span>
            </div>
            <div className="text-[11px] text-zinc-400 font-mono mt-0.5">Background Worker</div>
          </div>
          <div className="w-11 h-11 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400">
            <Clock className="w-5 h-5" />
          </div>
        </div>

        <div className="p-5 rounded-2xl border border-zinc-800/80 bg-zinc-950/70 backdrop-blur-md flex items-center justify-between">
          <div>
            <div className="text-xs font-mono text-zinc-500 uppercase tracking-wider">Geospatial Exports</div>
            <div className="text-base font-bold text-zinc-100 mt-1">
              {exportsCount} Products
            </div>
            <div className="text-[11px] text-zinc-400 font-mono mt-0.5">GeoTIFF / SAM QA</div>
          </div>
          <div className="w-11 h-11 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <HardDrive className="w-5 h-5" />
          </div>
        </div>
      </div>

      {/* Guided 4-Step Workflow Map */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              Guided Workflow
            </h2>
            <p className="text-xs text-zinc-400 font-sans mt-0.5">
              Follow the end-to-end analytical pipeline from raw scene ingestion to geospatial delivery.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5">
          {WORKFLOW_STEPS.map((step) => {
            const Icon = step.icon;
            return (
              <Link
                key={step.step}
                href={step.href}
                className="group rounded-2xl border border-zinc-800/80 bg-zinc-950/80 p-6 flex flex-col justify-between transition-all hover:border-amber-500/50 hover:bg-zinc-900/50 hover:shadow-xl hover:shadow-amber-500/5"
              >
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <span className="font-mono text-xs font-bold text-zinc-500 group-hover:text-amber-400 transition-colors">
                      STEP {step.step}
                    </span>
                    <span className="px-2 py-0.5 rounded font-mono text-[10px] font-semibold bg-zinc-900 border border-zinc-800 text-zinc-300">
                      {step.badge}
                    </span>
                  </div>
                  <div className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-zinc-300 group-hover:text-amber-400 group-hover:border-amber-500/40 transition-colors mb-3">
                    <Icon className="w-5 h-5" />
                  </div>
                  <h3 className="text-lg font-bold text-zinc-100 group-hover:text-white transition-colors">
                    {step.title}
                  </h3>
                  <p className="text-xs text-zinc-400 mt-2 leading-relaxed">
                    {step.desc}
                  </p>
                </div>

                <div className="pt-5 mt-4 border-t border-zinc-850 flex items-center justify-between text-xs font-mono font-medium text-zinc-400 group-hover:text-amber-300 transition-colors">
                  <span>Open workspace</span>
                  <ArrowRight className="w-4 h-4 transform group-hover:translate-x-1 transition-transform" />
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Analytical Workspaces Grid */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white tracking-tight">
              Analytical Workspaces
            </h2>
            <p className="text-xs text-zinc-400 font-sans mt-0.5">
              Specialized scientific investigation panels operating on 2.5m super-resolved products.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {QUICK_WORKSPACES.map((ws) => {
            const Icon = ws.icon;
            return (
              <Link
                key={ws.href}
                href={ws.href}
                className="group p-5 rounded-2xl border border-zinc-800/80 bg-zinc-950/70 hover:bg-zinc-900/60 hover:border-zinc-700 transition flex items-start gap-4"
              >
                <div className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-zinc-400 group-hover:text-amber-400 group-hover:border-amber-500/40 transition-colors shrink-0">
                  <Icon className="w-5 h-5" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-bold text-zinc-200 group-hover:text-white transition-colors truncate">
                      {ws.name}
                    </h3>
                    <ArrowRight className="w-3.5 h-3.5 text-zinc-500 group-hover:text-amber-400 group-hover:translate-x-0.5 transition-all shrink-0 ml-2" />
                  </div>
                  <p className="text-xs text-zinc-400 mt-1 leading-relaxed">
                    {ws.desc}
                  </p>
                </div>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}
