"use client";

import React from "react";
import { WorkspaceShell } from "@/components/console/WorkspaceShell";
import { ExportPanel } from "@/features/export/components/ExportPanel";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { Download } from "lucide-react";
import { useConsoleStore } from "@/lib/store";

export default function ExportWorkspacePage() {
  const currentRunId = useConsoleStore((s) => s.currentRunId);

  return (
    <WorkspaceShell
      title="Geospatial Products &amp; Exports"
      description="Download georeferenced Float32 GeoTIFF rasters, verified QA validation metrics, and analytical masks formatted for QGIS, ArcGIS, and GDAL workflows."
      icon={Download}
      status={
        currentRunId ? (
          <span className="px-2.5 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-700/60 text-emerald-300 font-mono text-xs uppercase">
            Active Run: {currentRunId.slice(-8)}
          </span>
        ) : (
          <span className="px-2.5 py-0.5 rounded-full bg-zinc-900 border border-zinc-800 text-zinc-400 font-mono text-xs uppercase">
            Global Product Archive
          </span>
        )
      }
    >
      <ErrorBoundary fallbackTitle="Export Panel Error">
        <ExportPanel />
      </ErrorBoundary>
    </WorkspaceShell>
  );
}
