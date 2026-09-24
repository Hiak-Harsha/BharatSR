/**
 * BharatSR API Client (Phase 6 & 7)
 * Interfaces with FastAPI backend at http://127.0.0.1:8000
 */

// In browser environments, use relative URLs ("") so Next.js rewrites proxy to backend seamlessly.
// In SSR or when NEXT_PUBLIC_API_URL is explicitly set, use that URL.
const API_BASE =
  process.env.NEXT_PUBLIC_API_URL !== undefined
    ? process.env.NEXT_PUBLIC_API_URL
    : (typeof window !== "undefined" ? "" : "http://127.0.0.1:8000");

export interface ModelInfo {
  id: string;
  name: string;
  architecture: string;
  scale_factor: number;
  n_bands: number;
  status: string;
  has_uncertainty?: boolean;
  checkpoint_path?: string;
  description?: string;
}

export interface MultiSpectralViews {
  rgb?: string;   // True Color
  cir?: string;   // Color Infrared (NIR, Red, Green)
  ndvi?: string;  // Normalized Difference Vegetation Index
  red?: string;   // Band 4 (Red)
  green?: string; // Band 3 (Green)
  blue?: string;  // Band 2 (Blue)
  nir?: string;   // Band 8 (NIR)
  error?: string; // Absolute Error Map |SR - HR|
}

export interface SampleTile {
  id: string;
  filename: string;
  title?: string;
  region?: string;
  description?: string;
  tactical_category?: string;
  coordinates?: string;
  bands: number;
  lr_size: string;
  has_ground_truth: boolean;
  thumbnail: string;
  views?: MultiSpectralViews;
}

export interface PixelBandData {
  name: string;
  band: string;
  wavelength: string;
  lr_reflectance: number;
  bicubic_reflectance?: number;
  sr_reflectance: number;
  hr_reflectance: number | null;
}

export interface PixelProfileResponse {
  status: string;
  sample_id?: string;
  run_id?: string;
  model_id: string;
  hr_coordinates: { x: number; y: number };
  lr_coordinates: { x: number; y: number };
  bands_data: PixelBandData[];
  ndvi: {
    sr: number;
    lr: number;
    bicubic?: number;
    hr: number | null;
  };
  spectral_angle_deg?: number;
  surface_classification: string;
  signature_analysis: string;
  interpretation_disclaimer?: string;
}

export interface DownstreamTaskMetrics {
  f1: number;
  iou: number;
  precision: number;
  recall: number;
}

export interface DownstreamTaskItem {
  task_name: string;
  description: string;
  bicubic: DownstreamTaskMetrics;
  rcan: DownstreamTaskMetrics;
  ground_truth_pixel_count: number;
  masks: {
    bicubic: string;
    rcan: string;
    ground_truth?: string;
  };
}

export interface DownstreamMasksResponse {
  status: string;
  tasks: {
    canopy_segmentation: DownstreamTaskItem;
    built_up_infrastructure: DownstreamTaskItem;
    [key: string]: DownstreamTaskItem;
  };
}


export interface MetricItem {
  value: number | null;
  unit: string;
  quality?: string | null;
  description: string;
}

export interface SuperResolveResponse {
  status: string;
  model_id: string;
  run_id?: string;
  /** "fast" (single pass) or "high" (4x flip self-ensemble TTA) — see run_inference_ensembled. */
  quality?: "fast" | "high";
  inference_time_s: number;
  input: {
    shape: number[];
    image: string; // base64 data URI
    views?: MultiSpectralViews;
  };
  bicubic?: {
    shape: number[];
    image: string; // base64 data URI
    views?: MultiSpectralViews;
  };
  output: {
    shape: number[];
    image: string; // base64 data URI
    views?: MultiSpectralViews;
  };
  error_map?: {
    image: string;
    mean_error: number;
    max_error: number;
  };
  ground_truth?: {
    shape: number[];
    image: string; // base64 data URI
    views?: MultiSpectralViews;
  };
  geospatial_metadata?: any;
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
    sr_stats?: {
      min: number;
      max: number;
      mean: number;
      bright_pixels_pct?: number;
    };
  };
  uncertainty?: {
    image: string; // base64 data URI of heatmap
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
}

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
}

export interface ComparisonTableRow {
  metric: string;
  unit: string;
  higher_is_better: boolean;
  best_model?: string;
  bicubic?: number | null;
  srcnn?: number | null;
  rcan?: number | null;
}

export interface CompareResponse {
  status: string;
  run_id?: string;
  input: {
    shape: number[];
    image: string;
    views?: MultiSpectralViews;
  };
  ground_truth?: {
    shape: number[];
    image: string;
    views?: MultiSpectralViews;
  };
  models: Record<string, CompareModelResult>;
  comparison_table: ComparisonTableRow[];
}

export interface JobStatusResponse {
  job_id: string;
  status: "pending" | "processing" | "completed" | "failed";
  model_id: string;
  progress_pct: number;
  is_cancelled: boolean;
  created_at: string;
  completed_at?: string;
  inference_time_s?: number;
  error_message?: string;
  result?: any;
}

export async function fetchHealth(): Promise<{ status: string; models_loaded: number; device: string; version: string; active_jobs: number }> {
  const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Backend offline: ${res.statusText}`);
  return res.json();
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE}/api/models`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`);
  const data = await res.json();
  return data.models;
}

export async function fetchSamples(): Promise<SampleTile[]> {
  const res = await fetch(`${API_BASE}/api/samples`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch samples: ${res.statusText}`);
  const data = await res.json();
  return data.samples;
}

export const checkHealth = fetchHealth;
export const getModels = fetchModels;
export const getSamples = fetchSamples;

export async function superresolveSample(
  sampleId: string,
  modelId: string = "rcan",
  quality: "fast" | "high" = "fast"
): Promise<SuperResolveResponse> {
  const formData = new FormData();
  formData.append("sample_id", sampleId);
  formData.append("model_id", modelId);
  formData.append("quality", quality);

  const res = await fetch(`${API_BASE}/api/superresolve`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Super-resolution failed");
  }

  return res.json();
}

export async function superresolveUpload(
  file: File,
  modelId: string = "rcan",
  quality: "fast" | "high" = "fast"
): Promise<SuperResolveResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("model_id", modelId);
  formData.append("quality", quality);

  const res = await fetch(`${API_BASE}/api/superresolve`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Super-resolution failed");
  }

  return res.json();
}

/** Small live-inference crop (LR / Bicubic / BharatSR / Ground Truth) for the landing page. */
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

export async function getSamplePreview(sampleId: string, crop: number = 32): Promise<SamplePreviewResponse> {
  const res = await fetch(`${API_BASE}/api/samples/${encodeURIComponent(sampleId)}/preview?crop=${crop}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Failed to fetch preview: ${res.statusText}`);
  return res.json();
}

export async function compareModels(
  sampleId?: string,
  file?: File
): Promise<CompareResponse> {
  const formData = new FormData();
  if (sampleId) formData.append("sample_id", sampleId);
  if (file) formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/compare`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Model comparison failed");
  }

  return res.json();
}

export async function submitAsyncSuperresolve(
  sampleId?: string,
  file?: File,
  modelId: string = "rcan",
  quality: "fast" | "high" = "fast"
): Promise<{ status: string; job_id: string; status_url: string }> {
  const formData = new FormData();
  if (sampleId) formData.append("sample_id", sampleId);
  if (file) formData.append("file", file);
  formData.append("model_id", modelId);
  formData.append("quality", quality);

  const res = await fetch(`${API_BASE}/api/superresolve/async`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Async job submission failed");
  }

  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to get job status: ${res.statusText}`);
  return res.json();
}

export async function getJobs(): Promise<any[]> {
  const res = await fetch(`${API_BASE}/api/jobs`, { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to fetch jobs: ${res.statusText}`);
  const data = await res.json();
  return data.jobs || [];
}

export function getGeoTIFFDownloadUrl(sampleId?: string, runId?: string, modelId: string = "rcan"): string {
  if (runId) {
    return `${API_BASE}/api/export/geotiff?run_id=${encodeURIComponent(runId)}&model_id=${encodeURIComponent(modelId)}`;
  }
  return `${API_BASE}/api/export/geotiff?sample_id=${encodeURIComponent(sampleId || "sample_real_s2")}&model_id=${encodeURIComponent(modelId)}`;
}

export function getReportDownloadUrl(sampleId?: string, modelId: string = "rcan"): string {
  return `${API_BASE}/api/export/report?sample_id=${encodeURIComponent(sampleId || "sample_real_s2")}&model_id=${encodeURIComponent(modelId)}`;
}

export async function getPixelProfile(
  sampleId?: string,
  runId?: string,
  x: number = 128,
  y: number = 128,
  modelId: string = "rcan"
): Promise<PixelProfileResponse> {
  const formData = new FormData();
  if (runId) {
    formData.append("run_id", runId);
  } else if (sampleId) {
    formData.append("sample_id", sampleId);
  }
  formData.append("x", x.toString());
  formData.append("y", y.toString());
  formData.append("model_id", modelId);

  const res = await fetch(`${API_BASE}/api/pixel-profile`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Pixel profiling failed");
  }

  return res.json();
}

export async function getDownstreamMasks(
  sampleId?: string,
  runId?: string,
  modelId: string = "rcan"
): Promise<DownstreamMasksResponse> {
  const formData = new FormData();
  if (runId) {
    formData.append("run_id", runId);
  } else if (sampleId) {
    formData.append("sample_id", sampleId);
  }
  formData.append("model_id", modelId);

  const res = await fetch(`${API_BASE}/api/downstream-masks`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Downstream task segmentation failed");
  }

  return res.json();
}
