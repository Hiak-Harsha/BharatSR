"use client";

import React from "react";
import { WorkspaceShell } from "@/components/console/WorkspaceShell";
import { InferencePanel } from "@/features/inference/components/InferencePanel";
import { ModelSelector } from "@/features/models/components/ModelSelector";
import { SampleGallery } from "@/features/samples/components/SampleGallery";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { Sparkles, Layers } from "lucide-react";
import { useConsoleStore } from "@/lib/store";

export default function RunWorkspacePage() {
  const currentRunId = useConsoleStore((s) => s.currentRunId);
  const selectedModel = useConsoleStore((s) => s.selectedModel);

  return (
    <WorkspaceShell
      title="Super-Resolution Inference"
      description="Execute 4× deep neural network super-resolution on Sentinel-2 10m reflectance tiles to generate 2.5m analytical surface products with physics constraints."
      icon={Sparkles}
      status={
        currentRunId ? (
          <span className="px-2.5 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-700/60 text-emerald-300 font-mono text-xs uppercase">
            Run Active: {currentRunId.slice(-8)}
          </span>
        ) : (
          <span className="px-2.5 py-0.5 rounded-full bg-zinc-900 border border-zinc-800 text-zinc-400 font-mono text-xs uppercase">
            Ready for inference ({selectedModel.toUpperCase()})
          </span>
        )
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
      <ErrorBoundary fallbackTitle="Inference Panel Error">
        <InferencePanel />
      </ErrorBoundary>
    </WorkspaceShell>
  );
}
