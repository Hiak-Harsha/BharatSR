import { useMutation } from "@tanstack/react-query";
import { getFieldBoundary, FieldBoundaryResponse } from "@/lib/api-client";

interface FieldBoundaryParams {
  sampleId?: string;
  runId?: string;
  modelId?: string;
  method?: string;
}

export function useFieldBoundary() {
  return useMutation<FieldBoundaryResponse, Error, FieldBoundaryParams>({
    mutationFn: ({ sampleId, runId, modelId, method }) =>
      getFieldBoundary(sampleId, runId, modelId, method),
  });
}
