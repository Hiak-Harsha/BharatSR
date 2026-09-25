import { useQuery } from "@tanstack/react-query";
import { fetchDatasetSummary, DatasetSummaryResponse } from "@/lib/api-client";

export function useDatasetSummary() {
  return useQuery<DatasetSummaryResponse, Error>({
    queryKey: ["datasetSummary"],
    queryFn: fetchDatasetSummary,
    staleTime: 60000,
  });
}
