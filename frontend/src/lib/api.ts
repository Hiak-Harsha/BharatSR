/**
 * BharatSR API Re-exports & Compatibility Facade
 * Delegates to src/lib/api-client.ts (generated from OpenAPI spec).
 */

export * from "./api-client";
import * as client from "./api-client";
import type { MultiSpectralViews, MetricItem } from "./api-client";

export const checkHealth = client.fetchHealth;
export const getModels = client.fetchModels;
export const getSamples = client.fetchSamples;

export interface CompareModelResult {
  model_id: string;
  inference_time_s: number;
  shape: number[];
  image: string;
  views?: MultiSpectralViews;
  metrics: {
    psnr: MetricItem;
    ssim: MetricItem;
    sam: MetricItem;
    downsample_consistency: MetricItem;
    spectral_mae?: MetricItem;
    gradient_similarity?: MetricItem;
    hallucination_fidelity?: {
      false_edge_rate: number;
      missing_edge_rate: number;
      high_freq_hallucination_rate: number;
      correctness_score: number;
      consistency_score: number;
      synthesis_score: number;
    };
    [key: string]: any;
  };
  uncertainty?: {
    image: string;
    summary: {
      mean_sigma: number;
      std_sigma: number;
      min_sigma: number;
      max_sigma: number;
      high_uncertainty_fraction: number;
      calibration_status?: string;
    };
    scatter?: {
      correlation: number;
      points: Array<{ unc: number; err: number }>;
    };
  };
  [key: string]: any;
}

export interface ComparisonTableRow {
  metric: string;
  unit: string;
  higher_is_better: boolean;
  best_model?: string;
  bicubic?: number | null;
  srcnn?: number | null;
  rcan?: number | null;
  [key: string]: any;
}

export async function superresolveSample(
  sampleId: string,
  modelId: string = "rcan",
  quality: "fast" | "high" = "fast"
) {
  return client.superresolve({ sampleId, modelId, quality });
}

export async function superresolveUpload(
  file: File,
  modelId: string = "rcan",
  quality: "fast" | "high" = "fast"
) {
  return client.superresolve({ file, modelId, quality });
}
