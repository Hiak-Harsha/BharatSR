"use client";

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getJobs, cancelJob, submitBatch, JobStatusResponse } from "@/lib/api-client";
import { useSamples } from "@/features/samples/hooks/useSamples";
import { useConsoleStore } from "@/lib/store";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  ListFilter,
  Play,
  RefreshCw,
  XCircle,
  Eye,
  CheckCircle2,
  AlertCircle,
  Layers,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

export function BatchJobQueue({ className }: { className?: string }) {
  const { data: samples, isLoading: isLoadingSamples } = useSamples();
  const [selectedIds, setSelectedIds] = useState<string[]>(["sample_real_s2"]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submissionFeedback, setSubmissionFeedback] = useState<string | null>(null);

  const setCurrentRunId = useConsoleStore((s) => s.setCurrentRunId);
  const setSelectedSample = useConsoleStore((s) => s.setSelectedSample);
  const hydrateSession = useConsoleStore((s) => s.hydrateSession);
  const setActiveView = useConsoleStore((s) => s.setActiveView);
  const selectedModel = useConsoleStore((s) => s.selectedModel);

  const {
    data: jobs,
    isLoading: isLoadingJobs,
    refetch,
  } = useQuery<JobStatusResponse[]>({
    queryKey: ["jobs"],
    queryFn: getJobs,
    refetchInterval: 3000,
  });

  const toggleSample = (sampleId: string) => {
    setSelectedIds((prev) =>
      prev.includes(sampleId) ? prev.filter((id) => id !== sampleId) : [...prev, sampleId]
    );
  };

  const handleBatchSubmit = async () => {
    if (selectedIds.length === 0) return;
    setIsSubmitting(true);
    setSubmissionFeedback(null);
    try {
      const res = await submitBatch(selectedIds.join(","), undefined, selectedModel, "fast");
      setSubmissionFeedback(`Batch ${res.batch_id.slice(0, 8)} queued: ${res.total} scene(s) submitted.`);
      refetch();
    } catch (err: any) {
      setSubmissionFeedback(`Batch submission failed: ${err.message || err}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleCancel = async (jobId: string) => {
    try {
      await cancelJob(jobId);
      refetch();
    } catch (err) {
      console.error("Cancel job failed:", err);
    }
  };

  const handleViewResult = (job: JobStatusResponse) => {
    const runId = (job.result as any)?.run_id || job.job_id;
    const sampleId = (job as any).sample_id;
    setCurrentRunId(runId);
    if (sampleId) {
      setSelectedSample(sampleId);
    }
    hydrateSession({
      runId,
      jobId: job.job_id,
      sampleId: sampleId || undefined,
      modelId: job.model_id,
      status: "completed",
      metrics: (job.result as any)?.metrics || {},
      uncertainty: (job.result as any)?.uncertainty || undefined,
    });
    setActiveView("single");
  };

  // Compute batch statistics
  const totalJobs = jobs?.length || 0;
  const queuedCount = jobs?.filter((j) => j.status === "pending" || j.status === "queued").length || 0;
  const processingCount = jobs?.filter((j) => j.status === "processing" || j.status === "running").length || 0;
  const completedCount = jobs?.filter((j) => j.status === "completed").length || 0;
  const failedCount = jobs?.filter((j) => j.status === "failed" || j.is_cancelled).length || 0;

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      {/* Batch Ingestion & Scene Selection Section */}
      <div className="p-5 rounded-xl border border-zinc-800 bg-zinc-950 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-zinc-850 pb-3">
          <div>
            <h2 className="font-mono text-base font-bold text-zinc-100 flex items-center gap-2">
              <ListFilter className="w-5 h-5 text-amber-400" />
              Batch Scene Ingestion & Background Worker
            </h2>
            <p className="mt-1 font-mono text-xs text-zinc-400">
              Select available scenes to execute asynchronous super-resolution runs in the background.
            </p>
          </div>
          <button
            type="button"
            disabled={isSubmitting || selectedIds.length === 0}
            onClick={handleBatchSubmit}
            className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg font-mono text-xs font-bold uppercase bg-amber-500 hover:bg-amber-400 disabled:opacity-50 disabled:hover:bg-amber-500 text-zinc-950 transition shadow-lg shadow-amber-500/10 shrink-0"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            Queue Selected Scenes ({selectedIds.length})
          </button>
        </div>

        {/* Scene Selection Checklist */}
        <div className="space-y-2">
          <div className="font-mono text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center justify-between">
            <span>Available Scenes</span>
            <span className="text-zinc-500">{selectedIds.length} of {samples?.length || 0} selected</span>
          </div>

          {isLoadingSamples ? (
            <div className="space-y-2">
              <div className="h-10 bg-zinc-900 animate-pulse rounded-lg" />
              <div className="h-10 bg-zinc-900 animate-pulse rounded-lg" />
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {(samples as any[])?.map((s: any) => {
                const isChecked = selectedIds.includes(s.sample_id);
                return (
                  <label
                    key={s.sample_id}
                    onClick={() => toggleSample(s.sample_id)}
                    className={cn(
                      "flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition font-mono select-none",
                      isChecked
                        ? "bg-amber-500/10 border-amber-500/40 text-zinc-100"
                        : "bg-zinc-900/60 border-zinc-800/80 text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
                    )}
                  >
                    <input
                      type="checkbox"
                      checked={isChecked}
                      onChange={() => {}} // handled by parent div
                      className="w-4 h-4 rounded border-zinc-700 text-amber-500 focus:ring-0 focus:ring-offset-0 bg-zinc-800"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-bold truncate text-zinc-200">
                        {s.title}
                      </div>
                      <div className="text-[11px] text-zinc-400 flex items-center gap-2 mt-0.5">
                        <span>{s.sensor}</span>
                        <span>·</span>
                        <span>{s.gsd || "10m GSD"}</span>
                        <span>·</span>
                        <span className={s.is_independent_hr ? "text-emerald-400" : "text-zinc-500"}>
                          {s.is_independent_hr ? "Independent HR" : "Demo Reference"}
                        </span>
                      </div>
                    </div>
                  </label>
                );
              })}
            </div>
          )}
        </div>

        {submissionFeedback && (
          <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-3 font-mono text-xs text-amber-300 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0 text-amber-400" />
            <span>{submissionFeedback}</span>
          </div>
        )}
      </div>

      {/* Batch Overview Telemetry Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 font-mono">
        <div className="p-3 rounded-xl border border-zinc-800 bg-zinc-950">
          <div className="text-zinc-400 text-xs">Total Jobs</div>
          <div className="text-lg font-bold text-zinc-100 mt-1">{totalJobs}</div>
        </div>
        <div className="p-3 rounded-xl border border-zinc-800 bg-zinc-950">
          <div className="text-zinc-400 text-xs">Queued</div>
          <div className="text-lg font-bold text-amber-400 mt-1">{queuedCount}</div>
        </div>
        <div className="p-3 rounded-xl border border-zinc-800 bg-zinc-950">
          <div className="text-zinc-400 text-xs">Processing</div>
          <div className="text-lg font-bold text-cyan-400 mt-1">{processingCount}</div>
        </div>
        <div className="p-3 rounded-xl border border-zinc-800 bg-zinc-950">
          <div className="text-zinc-400 text-xs">Completed</div>
          <div className="text-lg font-bold text-emerald-400 mt-1">{completedCount}</div>
        </div>
        <div className="p-3 rounded-xl border border-zinc-800 bg-zinc-950 col-span-2 sm:col-span-1">
          <div className="text-zinc-400 text-xs">Failed / Cancelled</div>
          <div className="text-lg font-bold text-rose-400 mt-1">{failedCount}</div>
        </div>
      </div>

      {/* Jobs Table */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
        <div className="flex items-center justify-between border-b border-zinc-850 pb-3 mb-3">
          <h3 className="font-mono text-sm font-semibold uppercase tracking-wider text-zinc-200">
            Execution Queue ({jobs?.length || 0})
          </h3>
          <button
            type="button"
            onClick={() => refetch()}
            className="inline-flex items-center gap-1.5 text-xs font-mono text-zinc-400 hover:text-zinc-200 transition"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Refresh Queue
          </button>
        </div>

        {isLoadingJobs ? (
          <div className="space-y-2">
            <div className="h-12 bg-zinc-900 animate-pulse rounded-lg" />
            <div className="h-12 bg-zinc-900 animate-pulse rounded-lg" />
          </div>
        ) : !jobs || jobs.length === 0 ? (
          <div className="p-8 text-center font-mono text-sm text-zinc-400">
            No active or recent jobs found in the queue.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead>
                <tr className="border-b border-zinc-850 text-zinc-400 text-xs">
                  <th className="py-3 px-3">Job ID</th>
                  <th className="py-3 px-3">Model</th>
                  <th className="py-3 px-3">Status</th>
                  <th className="py-3 px-3">Progress</th>
                  <th className="py-3 px-3">Created</th>
                  <th className="py-3 px-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-850">
                {jobs.map((job) => {
                  const isFinished = job.status === "completed" || job.status === "failed" || job.is_cancelled;
                  const isCompleted = job.status === "completed";
                  return (
                    <tr key={job.job_id} className="hover:bg-zinc-900/40">
                      <td className="py-3 px-3 font-semibold text-zinc-300">
                        {job.job_id.slice(0, 14)}...
                      </td>
                      <td className="py-3 px-3 text-zinc-300 uppercase font-medium">{job.model_id}</td>
                      <td className="py-3 px-3">
                        <div className="flex flex-col gap-1 items-start">
                          <StatusBadge
                            status={job.is_cancelled ? "cancelled" : job.status}
                            variant={
                              job.status === "completed"
                                ? "success"
                                : job.status === "failed"
                                ? "error"
                                : job.is_cancelled
                                ? "warning"
                                : "amber"
                            }
                          />
                          {(job.status === "failed" || job.error_message) && (
                            <div
                              className="text-[10px] font-sans text-rose-300 bg-rose-950/40 border border-rose-800/60 rounded px-1.5 py-0.5 mt-0.5 max-w-[220px] truncate"
                              title={job.error_message || "Inference worker error"}
                            >
                              {job.error_message || "Inference execution error"}
                            </div>
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-3 w-40">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-2 bg-zinc-800 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-amber-400 transition-all duration-300"
                              style={{ width: `${job.progress_pct || 0}%` }}
                            />
                          </div>
                          <span className="text-xs text-zinc-400 w-9 text-right">
                            {job.progress_pct || 0}%
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-3 text-zinc-400 text-xs">
                        {job.created_at ? new Date(job.created_at).toLocaleTimeString() : "—"}
                      </td>
                      <td className="py-3 px-3 text-right">
                        <div className="inline-flex items-center gap-2">
                          {isCompleted && (
                            <button
                              type="button"
                              onClick={() => handleViewResult(job)}
                              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 font-mono text-xs font-bold transition"
                            >
                              <Eye className="w-3.5 h-3.5" />
                              View Result
                            </button>
                          )}
                          {!isFinished && (
                            <button
                              type="button"
                              onClick={() => handleCancel(job.job_id)}
                              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded text-rose-400 hover:text-rose-300 hover:bg-rose-950/20 border border-rose-900/40 text-xs transition"
                            >
                              <XCircle className="w-3.5 h-3.5" />
                              Cancel
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
