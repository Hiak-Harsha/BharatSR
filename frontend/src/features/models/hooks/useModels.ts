import { useQuery } from "@tanstack/react-query";
import { fetchModels, ModelInfo } from "@/lib/api-client";

export function useModels() {
  return useQuery<ModelInfo[], Error>({
    queryKey: ["models"],
    queryFn: fetchModels,
    staleTime: 5 * 60 * 1000,
  });
}
