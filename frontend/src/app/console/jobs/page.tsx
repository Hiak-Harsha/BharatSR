"use client";

import React from "react";
import { WorkspaceShell } from "@/components/console/WorkspaceShell";
import { BatchJobQueue } from "@/features/jobs/components/BatchJobQueue";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { ListFilter } from "lucide-react";

export default function JobsWorkspacePage() {
  return (
    <WorkspaceShell
      title="Batch Processing &amp; Job Queue"
      description="Manage background inference workers, track asynchronous satellite tile reconstruction pipelines, and monitor execution throughput."
      icon={ListFilter}
      status={
        <span className="px-2.5 py-0.5 rounded-full bg-zinc-900 border border-zinc-800 text-zinc-300 font-mono text-xs uppercase">
          Worker: Active
        </span>
      }
    >
      <ErrorBoundary fallbackTitle="Batch Job Queue Error">
        <BatchJobQueue />
      </ErrorBoundary>
    </WorkspaceShell>
  );
}
