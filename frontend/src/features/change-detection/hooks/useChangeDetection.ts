import { useMutation } from "@tanstack/react-query";
import { getChangeDetection, ChangeDetectionResponse } from "@/lib/api-client";

interface ChangeDetectionParams {
  runIdT1: string;
  runIdT2: string;
  method?: string;
}

export function useChangeDetection() {
  return useMutation<ChangeDetectionResponse, Error, ChangeDetectionParams>({
    mutationFn: ({ runIdT1, runIdT2, method }) =>
      getChangeDetection(runIdT1, runIdT2, method),
  });
}
