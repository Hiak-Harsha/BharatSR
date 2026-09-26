/**
 * BharatSR Typed API Client
 * Generated & typed from backend OpenAPI specification.
 */

import { getApiBase, getWsBase } from "./env";
import type { components, paths } from "./api-schema";

// Type Aliases from OpenAPI components schema
export type ModelInfo = components["schemas"]["ModelInfo"];
export type SampleInfo = components["schemas"]["SampleInfo"];
export type SampleTile = SampleInfo; // Backward-compatibility alias
export type SuperResolveResponse = components["schemas"]["SuperResolveResponse"];
export type CompareResponse = components["schemas"]["CompareResponse"];
export type SpectralIndicesResponse = components["schemas"]["SpectralIndicesResponse"];
export type CropHealthResponse = components["schemas"]["CropHealthResponse"];
export type FieldBoundaryResponse = components["schemas"]["FieldBoundaryResponse"];
export type ChangeDetectionResponse = components["schemas"]["ChangeDetectionResponse"];
export type PixelProfileResponse = components["schemas"]["PixelProfileResponse"];
export type BandProfile = components["schemas"]["BandProfile"];
export type DownstreamMasksResponse = components["schemas"]["DownstreamMasksResponse"];
export type DownstreamTaskItem = components["schemas"]["DownstreamTaskItem"];
export type DownstreamTaskMetrics = components["schemas"]["DownstreamTaskMetrics"];
export type JobStatusResponse = components["schemas"]["JobStatusResponse"];
export type JobListResponse = components["schemas"]["JobListResponse"];
export type BatchSubmitResponse = components["schemas"]["BatchSubmitResponse"];
export type BatchStatusResponse = components["schemas"]["BatchStatusResponse"];
export type UncertaintyOutput = components["schemas"]["UncertaintyOutput"];
export type UncertaintySummary = components["schemas"]["UncertaintySummary"];
export type HealthResponse = components["schemas"]["HealthResponse"];
export type ReportResponse = components["schemas"]["ReportResponse"];
export type ModelReloadResponse = components["schemas"]["ModelReloadResponse"];
export type AsyncJobSubmitResponse = components["schemas"]["AsyncJobSubmitResponse"];
export type DatasetSummaryResponse = components["schemas"]["DatasetSummaryResponse"];
export type DatasetScenesResponse = components["schemas"]["DatasetScenesResponse"];
export type SceneRecord = components["schemas"]["SceneRecord"];
export type PreprocessingSampleResponse = components["schemas"]["PreprocessingSampleResponse"];
export type PreprocessingStageInfo = components["schemas"]["PreprocessingStageInfo"];
export type TrainingHistoryResponse = components["schemas"]["TrainingHistoryResponse"];
export type ModelCardResponse = components["schemas"]["ModelCardResponse"];
export type EpochMetric = components["schemas"]["EpochMetric"];

export interface MultiSpectralViews {
  composite?: string;
  rgb?: string;
  cir?: string;
  ndvi?: string;
  red?: string;
  green?: string;
  blue?: string;
  nir?: string;
  error?: string;
  [key: string]: string | undefined;
}

export interface MetricItem {
  value: number | null;
  unit: string;
  quality?: string | null;
  description: string;
}

export interface SamplePreviewResponse {
  status: string;
  sample_id: string;
  crop_size: number;
  model_id: string;
  views: {
    lr: string;
    bicubic: string;
    sr: string;
    ground_truth?: string;
  };
}

export interface NormalizedApiError {
  status: number;
  message: string;
  actionToFix: string;
  requestId?: string;
  endpoint: string;
  technicalDetails: string;
  validationErrors?: string[];
}

export class ApiError extends Error {
  status: number;
  detail: string;
  actionToFix: string;
  requestId?: string;
  endpoint: string;
  validationErrors?: string[];

  constructor(normalized: NormalizedApiError) {
    super(normalized.message);
    this.name = "ApiError";
    this.status = normalized.status;
    this.detail = normalized.message;
    this.actionToFix = normalized.actionToFix;
    this.requestId = normalized.requestId;
    this.endpoint = normalized.endpoint;
    this.validationErrors = normalized.validationErrors;
  }
}

export interface RequestOptions extends RequestInit {
  timeoutMs?: number;
}

function normalizeErrorResponse(
  status: number,
  body: any,
  endpoint: string,
  requestId?: string
): ApiError {
  let message = "An unexpected error occurred.";
  let actionToFix = "Retry the operation or contact technical support.";
  let validationErrors: string[] | undefined = undefined;

  if (status === 422 && body?.detail && Array.isArray(body.detail)) {
    const errors: string[] = body.detail.map((err: any) => {
      const field = Array.isArray(err.loc) ? err.loc.slice(1).join(".") : (err.loc || "Input");
      return `• ${field}: ${err.msg || "Invalid format"}`;
    });
    validationErrors = errors;
    message = `Input validation failed:\n${errors.join("\n")}`;
    actionToFix = "Ensure the uploaded Sentinel-2 GeoTIFF has all required bands (B2, B3, B4, B8) and valid georeferencing metadata.";
  } else if (status === 400) {
    message = body?.detail || body?.error || "Bad Request";
    if (message.includes("No geospatial reference")) {
      actionToFix = "GeoTIFF export requires georeferenced raster input. Synthetic benchmarks lack authentic CRS headers.";
    } else if (message.includes("Sentinel-2 model requires")) {
      actionToFix = "Provide a 4-band raster (B2 Blue, B3 Green, B4 Red, B8 NIR) or configure an explicit band mapping.";
    } else {
      actionToFix = "Review the selected options or verify input raster dimensions and format.";
    }
  } else if (status === 404) {
    message = body?.detail || body?.error || "Resource Not Found";
    actionToFix = "The requested scene, model, or run does not exist. Select an active run or available scene.";
  } else if (status === 408) {
    message = body?.detail || "Request timed out.";
    actionToFix = "Check the Jobs panel if this operation was submitted asynchronously, or select Fast quality mode.";
  } else if (status >= 500) {
    message = body?.detail || body?.error || "Internal Server Error";
    actionToFix = "The backend inference worker encountered an unexpected failure. Check server logs or retry with CPU mode.";
  } else if (body?.detail || body?.error) {
    message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || body.error);
  }

  const technicalDetails = `Status: ${status} | Endpoint: ${endpoint} | Request ID: ${requestId || "N/A"}`;

  return new ApiError({
    status,
    message,
    actionToFix,
    requestId,
    endpoint,
    technicalDetails,
    validationErrors,
  });
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const url = `${getApiBase()}${path}`;
  const headers = new Headers(options.headers || {});
  const timeoutMs = options.timeoutMs || 60000;

  // Inject API key if configured (from localStorage or environment variable)
  if (typeof window !== "undefined") {
    const key =
      localStorage.getItem("bharatsr_api_key") ||
      process.env.NEXT_PUBLIC_BHARATSR_API_KEY;
    if (key && !headers.has("X-API-Key")) {
      headers.set("X-API-Key", key);
    }
  }


  // Setup AbortController for category-specific timeouts
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  // If consumer passed an external signal, chain it
  if (options.signal) {
    options.signal.addEventListener("abort", () => controller.abort());
  }

  try {
    const res = await fetch(url, {
      ...options,
      headers,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    const requestId = res.headers.get("X-Request-ID") || undefined;

    if (!res.ok) {
      let body: any = null;
      try {
        body = await res.json();
      } catch {
        body = { detail: res.statusText };
      }
      throw normalizeErrorResponse(res.status, body, path, requestId);
    }

    return await res.json();
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err instanceof ApiError) {
      throw err;
    }
    if (err.name === "AbortError") {
      throw new ApiError({
        status: 408,
        message: "Request timed out while waiting for computation.",
        actionToFix: "If the backend is waking from a cold start, please retry in a moment.",
        endpoint: path,
        technicalDetails: `Timeout: ${timeoutMs}ms exceeded on ${path}`,
      });
    }
    const isNetworkFetchError =
      !err.status ||
      (typeof err.message === "string" && (err.message.includes("Failed to fetch") || err.message.includes("NetworkError"))) ||
      err.name === "TypeError";

    throw new ApiError({
      status: 0,
      message: isNetworkFetchError
        ? "Could not reach the analysis service. If this is the first request in a while, the backend may still be starting up — please try again in a moment."
        : (err.message || "Network connection failure."),
      actionToFix: "Verify that the BharatSR backend is reachable and click Retry.",
      endpoint: path,
      technicalDetails: String(err),
    });
  }
}

// -------------------------------------------------------------
// Health & Models
// -------------------------------------------------------------

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health", { cache: "no-store", timeoutMs: 60000 });
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const data = await request<components["schemas"]["ModelsListResponse"]>("/api/models", { cache: "no-store", timeoutMs: 60000 });
  return data.models;
}

export async function reloadModel(modelId: string, checkpointPath?: string): Promise<ModelReloadResponse> {
  const form = new FormData();
  form.append("model_id", modelId);
  if (checkpointPath) form.append("checkpoint_path", checkpointPath);
  return request<ModelReloadResponse>("/api/models/reload", {
    method: "POST",
    body: form,
    timeoutMs: 60000,
  });
}

// -------------------------------------------------------------
// Samples
// -------------------------------------------------------------

export async function fetchSamples(): Promise<SampleInfo[]> {
  const data = await request<components["schemas"]["SamplesListResponse"]>("/api/samples", { cache: "no-store", timeoutMs: 60000 });
  return data.samples;
}

export async function getSamplePreview(sampleId: string, crop: number = 32): Promise<SamplePreviewResponse> {
  return request<SamplePreviewResponse>(
    `/api/samples/${encodeURIComponent(sampleId)}/preview?crop=${crop}`,
    { cache: "no-store", timeoutMs: 60000 }
  );
}

// -------------------------------------------------------------
// Super-Resolution
// -------------------------------------------------------------

export interface SuperresolveParams {
  sampleId?: string;
  file?: File;
  modelId?: string;
  quality?: "fast" | "high";
}

export async function superresolve({
  sampleId,
  file,
  modelId = "rcan",
  quality = "fast",
}: SuperresolveParams): Promise<SuperResolveResponse> {
  const form = new FormData();
  if (file) {
    form.append("file", file);
  } else if (sampleId) {
    form.append("sample_id", sampleId);
  }
  form.append("model_id", modelId);
  form.append("quality", quality);

  // High quality 4-way TTA takes longer (up to 5 min timeout)
  const timeoutMs = quality === "high" ? 300000 : 120000;

  return request<SuperResolveResponse>("/api/superresolve", {
    method: "POST",
    body: form,
    timeoutMs,
  });
}

export async function submitAsyncSuperresolve({
  sampleId,
  file,
  modelId = "rcan",
  quality = "fast",
}: SuperresolveParams): Promise<AsyncJobSubmitResponse> {
  const form = new FormData();
  if (file) {
    form.append("file", file);
  } else if (sampleId) {
    form.append("sample_id", sampleId);
  }
  form.append("model_id", modelId);
  form.append("quality", quality);

  return request<AsyncJobSubmitResponse>("/api/superresolve/async", {
    method: "POST",
    body: form,
    timeoutMs: 60000,
  });
}

// -------------------------------------------------------------
// Model Comparison
// -------------------------------------------------------------

export async function compareModels(sampleId?: string, file?: File): Promise<CompareResponse> {
  const form = new FormData();
  if (sampleId) form.append("sample_id", sampleId);
  if (file) form.append("file", file);

  return request<CompareResponse>("/api/compare", {
    method: "POST",
    body: form,
    timeoutMs: 180000, // 3 min for multi-model execution
  });
}

// -------------------------------------------------------------
// Analysis Endpoints
// -------------------------------------------------------------

export async function getSpectralIndices(
  sampleId?: string,
  runId?: string,
  modelId: string = "rcan"
): Promise<SpectralIndicesResponse> {
  const form = new FormData();
  if (runId) form.append("run_id", runId);
  else if (sampleId) form.append("sample_id", sampleId);
  form.append("model_id", modelId);

  return request<SpectralIndicesResponse>("/api/indices", {
    method: "POST",
    body: form,
  });
}

export async function getCropHealth(
  sampleId?: string,
  runId?: string,
  modelId: string = "rcan"
): Promise<CropHealthResponse> {
  const form = new FormData();
  if (runId) form.append("run_id", runId);
  else if (sampleId) form.append("sample_id", sampleId);
  form.append("model_id", modelId);

  return request<CropHealthResponse>("/api/crop-health", {
    method: "POST",
    body: form,
  });
}

export async function getFieldBoundary(
  sampleId?: string,
  runId?: string,
  modelId: string = "rcan",
  method: string = "gradient"
): Promise<FieldBoundaryResponse> {
  const form = new FormData();
  if (runId) form.append("run_id", runId);
  else if (sampleId) form.append("sample_id", sampleId);
  form.append("model_id", modelId);
  form.append("method", method);

  return request<FieldBoundaryResponse>("/api/field-boundary", {
    method: "POST",
    body: form,
  });
}

export async function getChangeDetection(
  runIdT1: string,
  runIdT2: string,
  method: string = "ndvi_diff"
): Promise<ChangeDetectionResponse> {
  const form = new FormData();
  form.append("run_id_t1", runIdT1);
  form.append("run_id_t2", runIdT2);
  form.append("method", method);

  return request<ChangeDetectionResponse>("/api/change-detect", {
    method: "POST",
    body: form,
  });
}

export async function getPixelProfile(
  sampleId?: string,
  runId?: string,
  x: number = 128,
  y: number = 128,
  modelId: string = "rcan"
): Promise<PixelProfileResponse> {
  const form = new FormData();
  if (runId) form.append("run_id", runId);
  else if (sampleId) form.append("sample_id", sampleId);
  form.append("x", x.toString());
  form.append("y", y.toString());
  form.append("model_id", modelId);

  return request<PixelProfileResponse>("/api/pixel-profile", {
    method: "POST",
    body: form,
  });
}

export async function getDownstreamMasks(
  sampleId?: string,
  runId?: string,
  modelId: string = "rcan"
): Promise<DownstreamMasksResponse> {
  const form = new FormData();
  if (runId) form.append("run_id", runId);
  else if (sampleId) form.append("sample_id", sampleId);
  form.append("model_id", modelId);

  return request<DownstreamMasksResponse>("/api/downstream-masks", {
    method: "POST",
    body: form,
    timeoutMs: 120000,
  });
}

// -------------------------------------------------------------
// Jobs & Batch
// -------------------------------------------------------------

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`/api/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" });
}

export async function getJobs(): Promise<JobStatusResponse[]> {
  const data = await request<JobListResponse>("/api/jobs", { cache: "no-store" });
  return (data.jobs as unknown as JobStatusResponse[]) || [];
}

export async function cancelJob(jobId: string): Promise<{ status: string; job_id: string; message: string }> {
  return request<{ status: string; job_id: string; message: string }>(
    `/api/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: "POST" }
  );
}

export async function submitBatch(
  sampleIds?: string,
  files?: File[],
  modelId: string = "rcan",
  quality: string = "fast"
): Promise<BatchSubmitResponse> {
  const form = new FormData();
  if (sampleIds) form.append("sample_ids", sampleIds);
  if (files) {
    for (const f of files) form.append("files", f);
  }
  form.append("model_id", modelId);
  form.append("quality", quality);

  return request<BatchSubmitResponse>("/api/batch", {
    method: "POST",
    body: form,
  });
}

export async function getBatchStatus(batchId: string): Promise<BatchStatusResponse> {
  return request<BatchStatusResponse>(`/api/batch/${encodeURIComponent(batchId)}`, { cache: "no-store" });
}

// -------------------------------------------------------------
// Exports
// -------------------------------------------------------------

export function getGeoTIFFDownloadUrl(sampleId?: string, runId?: string, modelId: string = "rcan"): string {
  const base = getApiBase();
  if (runId) {
    return `${base}/api/export/geotiff?run_id=${encodeURIComponent(runId)}&model_id=${encodeURIComponent(modelId)}`;
  }
  return `${base}/api/export/geotiff?sample_id=${encodeURIComponent(sampleId || "sample_real_s2")}&model_id=${encodeURIComponent(modelId)}`;
}

export function getReportDownloadUrl(sampleId?: string, runId?: string, modelId: string = "rcan"): string {
  const base = getApiBase();
  if (runId) {
    return `${base}/api/export/report?run_id=${encodeURIComponent(runId)}&model_id=${encodeURIComponent(modelId)}`;
  }
  return `${base}/api/export/report?sample_id=${encodeURIComponent(sampleId || "sample_real_s2")}&model_id=${encodeURIComponent(modelId)}`;
}

export function getWebSocketInferenceUrl(jobId: string): string {
  return `${getWsBase()}/ws/inference/${encodeURIComponent(jobId)}`;
}

// -------------------------------------------------------------
// Dataset & Training Transparency (Part E)
// -------------------------------------------------------------

export async function fetchDatasetSummary(): Promise<DatasetSummaryResponse> {
  return request<DatasetSummaryResponse>("/api/dataset/summary", { cache: "no-store" });
}

export async function fetchDatasetScenes(
  split?: "train" | "val" | "test",
  region?: string,
  limit: number = 20,
  offset: number = 0
): Promise<DatasetScenesResponse> {
  const params = new URLSearchParams();
  if (split) params.append("split", split);
  if (region) params.append("region", region);
  params.append("limit", String(limit));
  params.append("offset", String(offset));
  return request<DatasetScenesResponse>(`/api/dataset/scenes?${params.toString()}`, { cache: "no-store" });
}

export function getQAReportUrl(split: "train" | "val" | "test"): string {
  return `${getApiBase()}/api/dataset/qa-report/${encodeURIComponent(split)}`;
}

export async function fetchPreprocessingSample(sceneId?: string): Promise<PreprocessingSampleResponse> {
  const query = sceneId ? `?scene_id=${encodeURIComponent(sceneId)}` : "";
  return request<PreprocessingSampleResponse>(`/api/dataset/preprocessing-sample${query}`, { cache: "no-store" });
}

export async function fetchTrainingHistory(modelName: string): Promise<TrainingHistoryResponse> {
  return request<TrainingHistoryResponse>(`/api/training/history/${encodeURIComponent(modelName)}`, { cache: "no-store" });
}

export async function fetchModelCard(modelName: string): Promise<ModelCardResponse> {
  return request<ModelCardResponse>(`/api/training/model-card/${encodeURIComponent(modelName)}`, { cache: "no-store" });
}

