"use client";

import React, { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getJobs, cancelJob, submitBatch, JobStatusResponse } from "@/lib/api-client";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { ListFilter, Play, RefreshCw, XCircle, Clock, CheckCircle, AlertTriangle } from "lucide-react";
import { formatTime, cn } from "@/lib/utils";

export function BatchJobQueue({ className }: { className?: string }) {
  const [selectedBatchSamples, setSelectedBatchSamples] = useState("sample_1,sample_2,sample_3");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const {
    data: jobs,
    isLoading,
    refetch,
  } = useQuery<JobStatusResponse[]>({
    queryKey: ["jobs"],
    queryFn: getJobs,
    refetchInterval: 3000, // Poll queue every 3s
  });

  const handleBatchSubmit = async () => {
    setIsSubmitting(true);
    try {
      await submitBatch(selectedBatchSamples);
      refetch();
    } catch (err) {
      console.error("Batch submit failed:", err);
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

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      {/* Batch Ingestion Card */}
      <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div>
          <h2 className="font-mono text-base font-bold text-zinc-100 flex items-center gap-2">
            <ListFilter className="w-5 h-5 text-amber-400" />
            Asynchronous Task & Batch Ingestion Queue
          </h2>
          <p className="mt-1 font-mono text-xs text-zinc-400">
            Submit multiple satellite tiles to the background ThreadPoolExecutor worker
          </p>
        </div>

        <div className="flex items-center gap-2 w-full md:w-auto">
          <input
            type="text"
            value={selectedBatchSamples}
            onChange={(e) => setSelectedBatchSamples(e.target.value)}
            placeholder="Comma-separated sample IDs"
            className="bg-zinc-900 border border-zinc-800 px-3 py-2 rounded-lg font-mono text-xs text-zinc-200 focus:outline-none focus:border-amber-500 w-full md:w-64"
          />
          <button
            type="button"
            disabled={isSubmitting || !selectedBatchSamples}
            onClick={handleBatchSubmit}
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg font-mono text-xs font-bold bg-amber-500 hover:bg-amber-400 text-zinc-950 shrink-0 transition"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            Queue Batch
          </button>
        </div>
      </div>

      {/* Jobs Table */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-4">
        <div className="flex items-center justify-between border-b border-zinc-850 pb-3 mb-3">
          <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-zinc-200">
            Active Job Queue ({jobs?.length || 0})
          </h3>
          <button
            type="button"
            onClick={() => refetch()}
            className="inline-flex items-center gap-1 text-xs font-mono text-zinc-400 hover:text-zinc-200"
          >
            <RefreshCw className="w-3 h-3" />
            Refresh
          </button>
        </div>

        {isLoading ? (
          <div className="space-y-2">
            <div className="h-12 bg-zinc-900 animate-pulse rounded" />
            <div className="h-12 bg-zinc-900 animate-pulse rounded" />
          </div>
        ) : !jobs || jobs.length === 0 ? (
          <div className="p-8 text-center font-mono text-xs text-zinc-500">
            No active or recent jobs found in the queue.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead>
                <tr className="border-b border-zinc-850 text-zinc-500 text-[11px]">
                  <th className="py-2.5 px-3">Job ID</th>
                  <th className="py-2.5 px-3">Model</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Progress</th>
                  <th className="py-2.5 px-3">Created</th>
                  <th className="py-2.5 px-3">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-850">
                {jobs.map((job) => {
                  const isFinished = job.status === "completed" || job.status === "failed" || job.is_cancelled;
                  return (
                    <tr key={job.job_id} className="hover:bg-zinc-900/40">
                      <td className="py-2.5 px-3 font-bold text-zinc-300">
                        {job.job_id.slice(0, 12)}...
                      </td>
                      <td className="py-2.5 px-3 text-zinc-400 uppercase">{job.model_id}</td>
                      <td className="py-2.5 px-3">
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
                      </td>
                      <td className="py-2.5 px-3 w-40">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-amber-400 transition-all duration-300"
                              style={{ width: `${job.progress_pct || 0}%` }}
                            />
                          </div>
                          <span className="text-[10px] text-zinc-500 w-8">
                            {job.progress_pct || 0}%
                          </span>
                        </div>
                      </td>
                      <td className="py-2.5 px-3 text-zinc-500 text-[11px]">
                        {job.created_at ? new Date(job.created_at).toLocaleTimeString() : "—"}
                      </td>
                      <td className="py-2.5 px-3">
                        {!isFinished && (
                          <button
                            type="button"
                            onClick={() => handleCancel(job.job_id)}
                            className="inline-flex items-center gap-1 text-[11px] text-rose-400 hover:text-rose-300"
                          >
                            <XCircle className="w-3.5 h-3.5" />
                            Cancel
                          </button>
                        )}
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
