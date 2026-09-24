import { useMutation } from "@tanstack/react-query";
import { getPixelProfile, PixelProfileResponse } from "@/lib/api-client";

interface PixelProfileParams {
  sampleId?: string;
  runId?: string;
  x: number;
  y: number;
  modelId?: string;
}

export function usePixelProfile() {
  return useMutation<PixelProfileResponse, Error, PixelProfileParams>({
    mutationFn: ({ sampleId, runId, x, y, modelId }) =>
      getPixelProfile(sampleId, runId, x, y, modelId),
  });
}
