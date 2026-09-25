"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/lib/api-client";
import { useConsoleStore, ConsoleViewMode } from "@/lib/store";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { StatusBadge } from "@/components/ui/StatusBadge";

// Feature Components
import { ModelSelector } from "@/features/models/components/ModelSelector";
import { SampleGallery } from "@/features/samples/components/SampleGallery";
import { InferencePanel } from "@/features/inference/components/InferencePanel";
import { CompareGrid } from "@/features/compare/components/CompareGrid";
import { SpectralIndicesDashboard } from "@/features/indices/components/SpectralIndicesDashboard";
import { CropHealthMap } from "@/features/crop-health/components/CropHealthMap";
import { FieldBoundaryOverlay } from "@/features/field-boundary/components/FieldBoundaryOverlay";
import { ChangeDetectionPanel } from "@/features/change-detection/components/ChangeDetectionPanel";
import { BatchJobQueue } from "@/features/jobs/components/BatchJobQueue";
import { ExportPanel } from "@/features/export/components/ExportPanel";
import { DownstreamMasksPanel } from "@/features/downstream/components/DownstreamMasksPanel";

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
  Server,
  Cpu,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_TABS: { id: ConsoleViewMode; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: "single", label: "Super-Resolution", icon: Sparkles },
  { id: "compare", label: "Model Comparison", icon: GitCompare },
  { id: "indices", label: "Spectral Indices", icon: BarChart3 },
  { id: "crop_health", label: "Crop Health", icon: Sprout },
  { id: "field_boundary", label: "Field Boundary", icon: Maximize2 },
  { id: "change_detection", label: "Change Detection", icon: History },
  { id: "downstream", label: "Downstream Tasks", icon: Target },
  { id: "jobs", label: "Batch Queue", icon: ListFilter },
  { id: "export", label: "Exports", icon: Download },
];

export default function ConsolePage() {
  const activeView = useConsoleStore((s) => s.activeView);
  const setActiveView = useConsoleStore((s) => s.setActiveView);
  const currentRunId = useConsoleStore((s) => s.currentRunId);

  const { data: health, isLoading: isHealthLoading, isError: isHealthError } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 10000,
  });

  return (
    <div className="flex flex-col gap-6">
      {/* Telemetry Status Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-3 bg-zinc-950/80 border border-zinc-800 rounded-xl backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 font-mono text-xs text-zinc-300">
            <Server className="w-4 h-4 text-emerald-400" />
            <span className="text-zinc-500 uppercase">Backend:</span>
            {isHealthLoading ? (
              <span className="text-zinc-500">Checking...</span>
            ) : isHealthError ? (
              <StatusBadge status="Offline" variant="error" />
            ) : (
              <StatusBadge status={health?.status || "Ready"} variant="success" />
            )}
          </div>

          <div className="hidden sm:flex items-center gap-1.5 font-mono text-xs text-zinc-400 pl-3 border-l border-zinc-800">
            <Cpu className="w-3.5 h-3.5 text-amber-400" />
            <span>Device:</span>
            <span className="text-zinc-200 uppercase font-semibold">
              {health?.device || "CPU"}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3 font-mono text-xs">
          {currentRunId && (
            <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded bg-zinc-900 border border-zinc-800 text-zinc-300">
              <span className="text-zinc-500">Active Run:</span>
              <span className="text-amber-400 font-bold">{currentRunId}</span>
            </div>
          )}
          <span className="text-zinc-500 text-[11px]">
            NTRO SIH26142 • Sentinel-2 4× Pipeline
          </span>
        </div>
      </div>

      {/* Main Feature Tabs Navigation */}
      <div className="flex flex-wrap items-center gap-1.5 p-1 bg-zinc-950/60 border border-zinc-800 rounded-xl overflow-x-auto">
        {NAV_TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeView === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveView(tab.id)}
              className={cn(
                "flex items-center gap-2 px-3.5 py-2 rounded-lg font-mono text-xs font-semibold tracking-wide transition-all whitespace-nowrap",
                isActive
                  ? "bg-amber-500/20 text-amber-300 border border-amber-500/50 shadow-[0_0_12px_rgba(245,158,11,0.2)]"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 border border-transparent"
              )}
            >
              <Icon className={cn("w-4 h-4", isActive ? "text-amber-400" : "text-zinc-500")} />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Main 2-Column Console Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Column: Model & Sample Selectors */}
        <div className="flex flex-col gap-6 lg:col-span-1">
          <ErrorBoundary fallbackTitle="Model Selector Component Error">
            <ModelSelector />
          </ErrorBoundary>

          <ErrorBoundary fallbackTitle="Sample Gallery Component Error">
            <SampleGallery />
          </ErrorBoundary>
        </div>

        {/* Right Column: Dynamic Feature Panel */}
        <div className="lg:col-span-3">
          {activeView === "single" && (
            <ErrorBoundary fallbackTitle="Inference Panel Error">
              <InferencePanel />
            </ErrorBoundary>
          )}

          {activeView === "compare" && (
            <ErrorBoundary fallbackTitle="Model Comparison Error">
              <CompareGrid />
            </ErrorBoundary>
          )}

          {activeView === "indices" && (
            <ErrorBoundary fallbackTitle="Spectral Indices Error">
              <SpectralIndicesDashboard />
            </ErrorBoundary>
          )}

          {activeView === "crop_health" && (
            <ErrorBoundary fallbackTitle="Crop Health Analysis Error">
              <CropHealthMap />
            </ErrorBoundary>
          )}

          {activeView === "field_boundary" && (
            <ErrorBoundary fallbackTitle="Field Boundary Delineation Error">
              <FieldBoundaryOverlay />
            </ErrorBoundary>
          )}

          {activeView === "change_detection" && (
            <ErrorBoundary fallbackTitle="Change Detection Error">
              <ChangeDetectionPanel />
            </ErrorBoundary>
          )}

          {activeView === "downstream" && (
            <ErrorBoundary fallbackTitle="Downstream Tasks Error">
              <DownstreamMasksPanel />
            </ErrorBoundary>
          )}

          {activeView === "jobs" && (
            <ErrorBoundary fallbackTitle="Batch Job Queue Error">
              <BatchJobQueue />
            </ErrorBoundary>
          )}

          {activeView === "export" && (
            <ErrorBoundary fallbackTitle="Exports Panel Error">
              <ExportPanel />
            </ErrorBoundary>
          )}
        </div>
      </div>
    </div>
  );
}
