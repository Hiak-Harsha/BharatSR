import { useQuery } from "@tanstack/react-query";
import { fetchSamples, SampleInfo } from "@/lib/api-client";

export function useSamples() {
  return useQuery<SampleInfo[], Error>({
    queryKey: ["samples"],
    queryFn: fetchSamples,
    staleTime: 5 * 60 * 1000,
  });
}
