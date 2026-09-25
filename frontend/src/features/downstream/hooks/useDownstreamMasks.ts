import { useQuery } from "@tanstack/react-query";
import { getDownstreamMasks, DownstreamMasksResponse } from "@/lib/api-client";

export function useDownstreamMasks(
  sampleId?: string,
  runId?: string,
  modelId: string = "rcan",
  enabled: boolean = true
) {
  return useQuery<DownstreamMasksResponse, Error>({
    queryKey: ["downstream-masks", sampleId, runId, modelId],
    queryFn: () => getDownstreamMasks(sampleId, runId, modelId),
    enabled: enabled && Boolean(sampleId || runId),
    staleTime: 5 * 60 * 1000,
  });
}
