import { useQuery } from "@tanstack/react-query";
import {
  fetchTrainingHistory,
  fetchModelCard,
  fetchPreprocessingSample,
  TrainingHistoryResponse,
  ModelCardResponse,
  PreprocessingSampleResponse,
} from "@/lib/api-client";

export function useTrainingHistory(modelName: string) {
  return useQuery<TrainingHistoryResponse, Error>({
    queryKey: ["trainingHistory", modelName],
    queryFn: () => fetchTrainingHistory(modelName),
    staleTime: 60000,
  });
}

export function useModelCard(modelName: string) {
  return useQuery<ModelCardResponse, Error>({
    queryKey: ["modelCard", modelName],
    queryFn: () => fetchModelCard(modelName),
    staleTime: 60000,
  });
}

export function usePreprocessingSample(sceneId?: string) {
  return useQuery<PreprocessingSampleResponse, Error>({
    queryKey: ["preprocessingSample", sceneId],
    queryFn: () => fetchPreprocessingSample(sceneId),
    staleTime: 60000,
  });
}
