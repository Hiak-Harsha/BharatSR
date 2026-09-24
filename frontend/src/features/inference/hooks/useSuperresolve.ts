import { useMutation } from "@tanstack/react-query";
import { superresolve, SuperresolveParams, SuperResolveResponse } from "@/lib/api-client";
import { useConsoleStore } from "@/lib/store";

export function useSuperresolve() {
  const setCurrentRunId = useConsoleStore((s) => s.setCurrentRunId);

  return useMutation<SuperResolveResponse, Error, SuperresolveParams>({
    mutationFn: (params) => superresolve(params),
    onSuccess: (data) => {
      if (data.run_id) {
        setCurrentRunId(data.run_id);
      }
    },
  });
}
