"use client";

import React from "react";
import { WorkspaceShell } from "@/components/console/WorkspaceShell";
import { DatasetTrainingPanel } from "@/features/dataset/components/DatasetTrainingPanel";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { Database } from "lucide-react";

export default function DatasetWorkspacePage() {
  return (
    <WorkspaceShell
      title="Dataset &amp; Training Transparency"
      description="Audit training data provenance, inspected 4-stage preprocessing walkthrough, QA validation reports, training convergence curves, and scientific model cards."
      icon={Database}
      status={
        <span className="px-2.5 py-0.5 rounded-full bg-amber-950/80 border border-amber-700/60 text-amber-300 font-mono text-xs uppercase">
          Copernicus Sentinel-2 L2A
        </span>
      }
    >
      <ErrorBoundary fallbackTitle="Dataset Transparency Panel Error">
        <DatasetTrainingPanel />
      </ErrorBoundary>
    </WorkspaceShell>
  );
}
