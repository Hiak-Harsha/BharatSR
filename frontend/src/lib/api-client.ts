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

export interface MultiSpectralViews {
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

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(`API Error ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const url = `${getApiBase()}${path}`;
  const headers = new Headers(options.headers || {});

  // Inject API key if configured
  if (typeof window !== "undefined") {
    const key = localStorage.getItem("bharatsr_api_key");
    if (key && !headers.has("X-API-Key")) {
      headers.set("X-API-Key", key);
    }
  }

  const res = await fetch(url, {
    ...options,
    headers,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || body.error || JSON.stringify(body);
    } catch {
      // ignore json parse error
    }
    throw new ApiError(res.status, detail);
  }

  return res.json();
}

// -------------------------------------------------------------
// Health & Models
// -------------------------------------------------------------

export async function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health", { cache: "no-store" });
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const data = await request<components["schemas"]["ModelsListResponse"]>("/api/models", { cache: "no-store" });
  return data.models;
}

export async function reloadModel(modelId: string, checkpointPath?: string): Promise<ModelReloadResponse> {
  const form = new FormData();
  form.append("model_id", modelId);
  if (checkpointPath) form.append("checkpoint_path", checkpointPath);
  return request<ModelReloadResponse>("/api/models/reload", {
    method: "POST",
    body: form,
  });
}

// -------------------------------------------------------------
// Samples
// -------------------------------------------------------------

export async function fetchSamples(): Promise<SampleInfo[]> {
  const data = await request<components["schemas"]["SamplesListResponse"]>("/api/samples", { cache: "no-store" });
  return data.samples;
}

export async function getSamplePreview(sampleId: string, crop: number = 32): Promise<SamplePreviewResponse> {
  return request<SamplePreviewResponse>(
    `/api/samples/${encodeURIComponent(sampleId)}/preview?crop=${crop}`,
    { cache: "no-store" }
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

  return request<SuperResolveResponse>("/api/superresolve", {
    method: "POST",
    body: form,
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
