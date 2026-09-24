import { useMutation } from "@tanstack/react-query";
import { getCropHealth, CropHealthResponse } from "@/lib/api-client";

interface CropHealthParams {
  sampleId?: string;
  runId?: string;
  modelId?: string;
}

export function useCropHealth() {
  return useMutation<CropHealthResponse, Error, CropHealthParams>({
    mutationFn: ({ sampleId, runId, modelId }) => getCropHealth(sampleId, runId, modelId),
  });
}
