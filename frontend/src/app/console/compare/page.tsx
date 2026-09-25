"use client";

import React from "react";
import { WorkspaceShell } from "@/components/console/WorkspaceShell";
import { CompareGrid } from "@/features/compare/components/CompareGrid";
import { ModelSelector } from "@/features/models/components/ModelSelector";
import { SampleGallery } from "@/features/samples/components/SampleGallery";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { GitCompare } from "lucide-react";
import { useConsoleStore } from "@/lib/store";

export default function CompareWorkspacePage() {
  const selectedSample = useConsoleStore((s) => s.selectedSample);

  return (
    <WorkspaceShell
      title="Model Comparison &amp; Ablation"
      description="Evaluate deep neural architectures side-by-side on identical input tiles: compare reconstruction clarity, spectral angle, and boundary sharpness."
      icon={GitCompare}
      status={
        <span className="px-2.5 py-0.5 rounded-full bg-amber-950/80 border border-amber-700/60 text-amber-300 font-mono text-xs uppercase">
          Scene: {selectedSample || "sample_real_s2"}
        </span>
      }
      sidebar={
        <>
          <ErrorBoundary fallbackTitle="Model Selection Error">
            <ModelSelector />
          </ErrorBoundary>
          <ErrorBoundary fallbackTitle="Sample Gallery Error">
            <SampleGallery />
          </ErrorBoundary>
        </>
      }
    >
      <ErrorBoundary fallbackTitle="Model Comparison Grid Error">
        <CompareGrid />
      </ErrorBoundary>
    </WorkspaceShell>
  );
}
