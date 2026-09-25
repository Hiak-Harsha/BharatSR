import { useQuery } from "@tanstack/react-query";
import { fetchDatasetScenes, DatasetScenesResponse } from "@/lib/api-client";

export function useDatasetScenes(
  split?: "train" | "val" | "test",
  region?: string,
  limit: number = 20,
  offset: number = 0
) {
  return useQuery<DatasetScenesResponse, Error>({
    queryKey: ["datasetScenes", split, region, limit, offset],
    queryFn: () => fetchDatasetScenes(split, region, limit, offset),
    staleTime: 30000,
  });
}
