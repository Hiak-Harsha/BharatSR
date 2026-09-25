"use client";

import React from "react";
import Link from "next/link";
import { WorkspaceShell } from "@/components/console/WorkspaceShell";
import { DownstreamMasksPanel } from "@/features/downstream/components/DownstreamMasksPanel";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { Target, Sparkles } from "lucide-react";
import { useConsoleStore } from "@/lib/store";

export default function DownstreamWorkspacePage() {
  const currentRunId = useConsoleStore((s) => s.currentRunId);
  const selectedSample = useConsoleStore((s) => s.selectedSample);

  return (
    <WorkspaceShell
      title="Downstream Semantic Segmentation"
      description="Validate downstream task performance gains: automated building footprint extraction, road network delineation, and waterbody masks from super-resolved scenes."
      icon={Target}
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
      <ErrorBoundary fallbackTitle="Downstream Tasks Error">
        <DownstreamMasksPanel />
      </ErrorBoundary>
    </WorkspaceShell>
  );
}
