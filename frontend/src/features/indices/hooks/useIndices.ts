import { useMutation } from "@tanstack/react-query";
import { getSpectralIndices, SpectralIndicesResponse } from "@/lib/api-client";

interface IndicesParams {
  sampleId?: string;
  runId?: string;
  modelId?: string;
}

export function useIndices() {
  return useMutation<SpectralIndicesResponse, Error, IndicesParams>({
    mutationFn: ({ sampleId, runId, modelId }) => getSpectralIndices(sampleId, runId, modelId),
  });
}
