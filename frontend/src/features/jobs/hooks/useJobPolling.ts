import { useQuery } from "@tanstack/react-query";
import { getJobStatus, JobStatusResponse } from "@/lib/api-client";

export function useJobPolling(jobId: string | null) {
  return useQuery<JobStatusResponse, Error>({
    queryKey: ["job", jobId],
    queryFn: () => getJobStatus(jobId!),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 1000;
      if (data.status === "completed" || data.status === "failed" || data.is_cancelled) {
        return false; // Stop polling on terminal state
      }
      return 1000; // Poll every 1s while pending/processing
    },
  });
}
