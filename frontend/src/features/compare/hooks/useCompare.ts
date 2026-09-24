import { useMutation } from "@tanstack/react-query";
import { compareModels, CompareResponse } from "@/lib/api-client";
import { useConsoleStore } from "@/lib/store";

interface CompareParams {
  sampleId?: string;
  file?: File;
}

export function useCompare() {
  const setCurrentRunId = useConsoleStore((s) => s.setCurrentRunId);

  return useMutation<CompareResponse, Error, CompareParams>({
    mutationFn: ({ sampleId, file }) => compareModels(sampleId, file),
    onSuccess: (data) => {
      if (data.run_id) {
        setCurrentRunId(data.run_id);
      }
    },
  });
}
