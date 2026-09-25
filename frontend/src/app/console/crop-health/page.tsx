"use client";

import React from "react";
import Link from "next/link";
import { WorkspaceShell } from "@/components/console/WorkspaceShell";
import { CropHealthMap } from "@/features/crop-health/components/CropHealthMap";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { Sprout, Sparkles } from "lucide-react";
import { useConsoleStore } from "@/lib/store";

export default function CropHealthWorkspacePage() {
  const currentRunId = useConsoleStore((s) => s.currentRunId);
  const selectedSample = useConsoleStore((s) => s.selectedSample);

  return (
    <WorkspaceShell
      title="Crop Health &amp; Canopy Assessment"
      description="Classify agricultural plots and forest canopies by photosynthetic vigor, leaf area index, and vegetation stress using 2.5m super-resolved NIR and Red spectral bands."
      icon={Sprout}
      status={
        currentRunId ? (
          <span className="px-2.5 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-700/60 text-emerald-300 font-mono text-xs uppercase">
            Run: {currentRunId.slice(-8)}
          </span>
        ) : (
          <span className="px-2.5 py-0.5 rounded-full bg-amber-950/80 border border-amber-700/60 text-amber-300 font-mono text-xs uppercase">
            Target: {selectedSample || "sample_real_s2"}
          </span>
        )
      }
      actions={
        !currentRunId ? (
          <Link
            href="/console/run"
            className="px-3.5 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-zinc-950 font-semibold text-xs font-mono transition flex items-center gap-1.5"
          >
            <Sparkles className="w-3.5 h-3.5" />
            Run Super-Resolution First
          </Link>
        ) : undefined
      }
    >
      <ErrorBoundary fallbackTitle="Crop Health Map Error">
        <CropHealthMap />
      </ErrorBoundary>
    </WorkspaceShell>
  );
}
