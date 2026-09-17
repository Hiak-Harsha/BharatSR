"use client";

import { useState, useEffect } from "react";
import ImageComparisonSlider from "@/components/ImageComparisonSlider";
import {
  checkHealth,
  getModels,
  getSamples,
  superresolveSample,
  superresolveUpload,
  compareModels,
  submitAsyncSuperresolve,
  getJobStatus,
  getJobs,
  getPixelProfile,
  getGeoTIFFDownloadUrl,
  getReportDownloadUrl,
  ModelInfo,
  SampleTile,
  SuperResolveResponse,
  CompareResponse,
  PixelProfileResponse,
} from "@/lib/api";

export default function Home() {
  const [backendStatus, setBackendStatus] = useState<"connecting" | "online" | "offline">("connecting");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("rcan");
  const [samples, setSamples] = useState<SampleTile[]>([]);
  const [selectedSample, setSelectedSample] = useState<string | null>(null);

  const [inputTab, setInputTab] = useState<"samples" | "upload">("samples");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const [loading, setLoading] = useState(false);
  const [loadingAction, setLoadingAction] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  // Results
  const [result, setResult] = useState<SuperResolveResponse | null>(null);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [activeCompareModel, setActiveCompareModel] = useState<string>("rcan");

  // Pixel Profile Inspector
  const [pixelProfile, setPixelProfile] = useState<PixelProfileResponse | null>(null);
  const [loadingPixel, setLoadingPixel] = useState(false);
  const [pixelCoord, setPixelCoord] = useState<{ x: number; y: number }>({ x: 128, y: 128 });

  // Async Jobs
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [jobStatusMsg, setJobStatusMsg] = useState<string | null>(null);
  const [recentJobs, setRecentJobs] = useState<any[]>([]);

  // Mission Briefing Modal
  const [showMissionModal, setShowMissionModal] = useState(false);

  // Initialize data on mount
  useEffect(() => {
    async function init() {
      try {
        await checkHealth();
        setBackendStatus("online");

        const [fetchedModels, fetchedSamples, fetchedJobs] = await Promise.all([
          getModels().catch(() => []),
          getSamples().catch(() => []),
          getJobs().catch(() => []),
        ]);

        setModels(fetchedModels);
        setSamples(fetchedSamples);
        setRecentJobs(fetchedJobs);

        if (fetchedSamples.length > 0) {
          setSelectedSample(fetchedSamples[0].id);
        }
      } catch (e) {
        console.error("Failed to connect to backend", e);
        setBackendStatus("offline");
      }
    }
    init();
  }, []);

  // Poll active async job
  useEffect(() => {
    if (!activeJobId) return;

    const interval = setInterval(async () => {
      try {
        const job = await getJobStatus(activeJobId);
        if (job.status === "completed" && job.result) {
          setResult(job.result);
          setJobStatusMsg(`✓ Job ${activeJobId} completed in ${job.inference_time_s}s`);
          setActiveJobId(null);
          getJobs().then(setRecentJobs).catch(() => {});
        } else if (job.status === "failed") {
          setError(`Job ${activeJobId} failed: ${job.error_message}`);
          setActiveJobId(null);
        } else {
          setJobStatusMsg(`Processing job ${activeJobId}... (${job.status})`);
        }
      } catch (e) {
        console.error("Job polling error", e);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [activeJobId]);

  // Inspect Pixel Profile
  const inspectPixel = async (sampleId: string, x: number, y: number, modelId: string = "rcan") => {
    setLoadingPixel(true);
    setPixelCoord({ x, y });
    try {
      const prof = await getPixelProfile(sampleId, x, y, modelId);
      setPixelProfile(prof);
    } catch (e) {
      console.error("Pixel inspection error:", e);
    } finally {
      setLoadingPixel(false);
    }
  };

  // Run Single Super-Resolution
  const handleProcess = async () => {
    setError(null);
    setLoading(true);
    setLoadingAction("inference");

    try {
      let res: SuperResolveResponse;
      if (inputTab === "samples" && selectedSample) {
        res = await superresolveSample(selectedSample, selectedModel);
        // Automatically inspect a central pixel to populate radiometric curve
        inspectPixel(selectedSample, 128, 128, selectedModel);
      } else if (inputTab === "upload" && uploadedFile) {
        res = await superresolveUpload(uploadedFile, selectedModel);
      } else {
        throw new Error("Please select a sample tile or upload an image.");
      }

      setResult(res);
      setCompareResult(null); // Switch to single view
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Inference failed";
      setError(message);
    } finally {
      setLoading(false);
      setLoadingAction("");
    }
  };

  // Run Multi-Model Comparison (Bicubic vs SRCNN vs RCAN)
  const handleCompare = async () => {
    setError(null);
    setLoading(true);
    setLoadingAction("compare");

    try {
      let res: CompareResponse;
      if (inputTab === "samples" && selectedSample) {
        res = await compareModels(selectedSample, undefined);
        inspectPixel(selectedSample, 128, 128, "rcan");
      } else if (inputTab === "upload" && uploadedFile) {
        res = await compareModels(undefined, uploadedFile);
      } else {
        throw new Error("Please select a sample tile or upload an image.");
      }

      setCompareResult(res);
      if (res.models["rcan"]) {
        setActiveCompareModel("rcan");
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Comparison failed";
      setError(message);
    } finally {
      setLoading(false);
      setLoadingAction("");
    }
  };

  // Submit Async Batch Job
  const handleAsyncSubmit = async () => {
    setError(null);
    setLoading(true);
    setLoadingAction("async");

    try {
      const res = await submitAsyncSuperresolve(
        inputTab === "samples" ? selectedSample || undefined : undefined,
        inputTab === "upload" ? uploadedFile || undefined : undefined,
        selectedModel
      );
      setActiveJobId(res.job_id);
      setJobStatusMsg(`Job ${res.job_id} submitted to queue...`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Submission failed";
      setError(message);
    } finally {
      setLoading(false);
      setLoadingAction("");
    }
  };

  // Active displayed result for the viewer
  const activeSR = compareResult && compareResult.models[activeCompareModel]
    ? {
        image: compareResult.models[activeCompareModel].image,
        views: compareResult.models[activeCompareModel].views,
        shape: compareResult.models[activeCompareModel].shape,
        uncertainty: compareResult.models[activeCompareModel].uncertainty,
        metrics: compareResult.models[activeCompareModel].metrics,
        model_id: activeCompareModel,
        inference_time_s: compareResult.models[activeCompareModel].inference_time_s,
      }
    : result
    ? {
        image: result.output.image,
        views: result.output.views,
        shape: result.output.shape,
        uncertainty: result.uncertainty,
        metrics: result.metrics,
        model_id: result.model_id,
        inference_time_s: result.inference_time_s,
      }
    : null;

  const activeInput = compareResult ? compareResult.input : result?.input;
  const activeGT = compareResult ? compareResult.ground_truth : result?.ground_truth;

  return (
    <div className="space-y-6">
      {/* System Status & Problem Statement Banner */}
      <div className="glass-panel p-4 flex flex-col md:flex-row md:items-center justify-between gap-4 border-slate-800">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                backendStatus === "online"
                  ? "bg-emerald-400 pulse-dot"
                  : backendStatus === "connecting"
                  ? "bg-amber-400"
                  : "bg-rose-500"
              }`}
            />
            <span className="text-slate-300 font-medium">
              API: {backendStatus === "online" ? "Online (127.0.0.1:8000)" : backendStatus}
            </span>
          </div>

          <div className="text-xs text-slate-400 font-mono hidden sm:block">
            Scale: <span className="text-cyan-400 font-semibold">4x Spatial</span> • Input: <span className="text-indigo-400 font-semibold">4-Band (RGB + NIR)</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowMissionModal(true)}
            className="px-3 py-1 rounded-lg bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 text-xs font-semibold hover:bg-indigo-500/30 transition flex items-center gap-1.5 shadow-sm"
          >
            <span>📑</span> NTRO SIH26142 Mission Briefing
          </button>
          <span className="px-2.5 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800/60 text-xs font-medium hidden lg:inline">
            Physics Compliant
          </span>
        </div>
      </div>

      {/* Main Grid: Left Controls & Right Viewer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Input Selection & Parameters (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* 1. Input Selector Panel */}
          <div className="glass-panel p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h2 className="text-sm font-semibold text-slate-200 tracking-wide uppercase">
                1. Select Satellite Imagery
              </h2>
              <div className="flex rounded-md bg-slate-900 p-0.5 text-xs">
                <button
                  onClick={() => setInputTab("samples")}
                  className={`px-3 py-1 rounded font-medium transition ${
                    inputTab === "samples"
                      ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Curated Tiles
                </button>
                <button
                  onClick={() => setInputTab("upload")}
                  className={`px-3 py-1 rounded font-medium transition ${
                    inputTab === "upload"
                      ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  Upload
                </button>
              </div>
            </div>

            {/* Presets Grid */}
            {inputTab === "samples" ? (
              <div className="space-y-3">
                <p className="text-xs text-slate-400">
                  Select a pre-calibrated multi-band satellite tile (Sentinel-2 / NAIP):
                </p>
                {samples.length === 0 ? (
                  <div className="text-center py-6 text-xs text-slate-500">
                    Loading sample tiles...
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-3">
                    {samples.map((s) => (
                      <button
                        key={s.id}
                        onClick={() => setSelectedSample(s.id)}
                        className={`p-2.5 rounded-xl border text-left transition flex flex-col gap-2 ${
                          selectedSample === s.id
                            ? "border-cyan-500 bg-cyan-950/30 ring-1 ring-cyan-500 shadow-md shadow-cyan-950/50"
                            : "border-slate-800 bg-slate-900/60 hover:border-slate-700 hover:bg-slate-900"
                        }`}
                      >
                        <div className="w-full aspect-video rounded-lg overflow-hidden bg-slate-950 border border-slate-800/80 relative">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={s.thumbnail}
                            alt={s.id}
                            className="w-full h-full object-cover"
                            style={{ imageRendering: "pixelated" }}
                          />
                          <span className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded bg-black/70 text-[9px] font-mono text-cyan-300">
                            {s.lr_size}
                          </span>
                        </div>
                        <div>
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-semibold text-slate-200">
                              {s.title || s.id.replace("_", " ")}
                            </span>
                            <span className="text-[10px] text-indigo-300 font-mono">
                              {s.bands}B
                            </span>
                          </div>
                          {s.region && (
                            <span className="text-[10px] text-cyan-400 font-mono block mt-0.5">
                              📍 {s.region}
                            </span>
                          )}
                          {s.tactical_category && (
                            <span className="text-[9px] text-slate-400 block mt-0.5 leading-tight line-clamp-1">
                              🎯 {s.tactical_category}
                            </span>
                          )}
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              /* Custom File Upload */
              <div className="space-y-3">
                <label className="block text-xs text-slate-400">
                  Upload Sentinel-2 GeoTIFF or Multi-Band Image (.tif, .png, .jpg):
                </label>
                <div className="border-2 border-dashed border-slate-800 rounded-xl p-6 text-center hover:border-cyan-500/50 transition bg-slate-900/40">
                  <input
                    type="file"
                    accept=".tif,.tiff,.png,.jpg,.jpeg,.npz"
                    onChange={(e) => setUploadedFile(e.target.files?.[0] || null)}
                    className="hidden"
                    id="file-upload"
                  />
                  <label htmlFor="file-upload" className="cursor-pointer space-y-2 block">
                    <svg className="w-8 h-8 mx-auto text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                    <span className="text-xs font-medium text-cyan-400 hover:text-cyan-300 block">
                      {uploadedFile ? uploadedFile.name : "Click to browse or drop tile"}
                    </span>
                    <span className="text-[10px] text-slate-500 block">
                      GeoTIFF (rasterio) or 4-band PNG supported
                    </span>
                  </label>
                </div>
              </div>
            )}
          </div>

          {/* 2. Model Selection Panel */}
          <div className="glass-panel p-5 space-y-4">
            <h2 className="text-sm font-semibold text-slate-200 tracking-wide uppercase border-b border-slate-800 pb-3">
              2. Neural Model Selection
            </h2>

            <div className="space-y-2.5">
              <label
                className={`p-3 rounded-xl border flex items-center justify-between cursor-pointer transition ${
                  selectedModel === "rcan"
                    ? "border-indigo-500 bg-indigo-950/25 ring-1 ring-indigo-500/50"
                    : "border-slate-800 bg-slate-900/40 hover:border-slate-700"
                }`}
              >
                <div className="flex items-center gap-3">
                  <input
                    type="radio"
                    name="model"
                    value="rcan"
                    checked={selectedModel === "rcan"}
                    onChange={() => setSelectedModel("rcan")}
                    className="accent-indigo-500"
                  />
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-slate-200">
                        RCAN Attention + Uncertainty
                      </span>
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 font-bold">
                        RECOMMENDED
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      Residual Channel Attention (RIR) with dual-head spatial variance
                    </div>
                  </div>
                </div>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-indigo-950 text-indigo-400 border border-indigo-800/60">
                  Dual-Head
                </span>
              </label>

              <label
                className={`p-3 rounded-xl border flex items-center justify-between cursor-pointer transition ${
                  selectedModel === "srcnn"
                    ? "border-cyan-500 bg-cyan-950/20"
                    : "border-slate-800 bg-slate-900/40"
                }`}
              >
                <div className="flex items-center gap-3">
                  <input
                    type="radio"
                    name="model"
                    value="srcnn"
                    checked={selectedModel === "srcnn"}
                    onChange={() => setSelectedModel("srcnn")}
                    className="accent-cyan-500"
                  />
                  <div>
                    <div className="text-xs font-semibold text-slate-200">
                      SRCNN Baseline
                    </div>
                    <div className="text-[11px] text-slate-400 mt-0.5">
                      3-layer CNN (L1 reflectance loss)
                    </div>
                  </div>
                </div>
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  Baseline
                </span>
              </label>
            </div>

            {/* Action Buttons */}
            <div className="space-y-2.5 pt-2">
              <button
                onClick={handleProcess}
                disabled={loading || (inputTab === "samples" && !selectedSample) || (inputTab === "upload" && !uploadedFile)}
                className={`w-full py-3.5 px-4 rounded-xl font-semibold text-sm transition shadow-lg flex items-center justify-center gap-2 ${
                  loading && loadingAction === "inference"
                    ? "bg-slate-800 text-slate-400 cursor-not-allowed"
                    : "bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white shadow-cyan-500/20 active:scale-[0.99]"
                }`}
              >
                {loading && loadingAction === "inference" ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-cyan-400" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing 4x Super-Resolution...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4 text-cyan-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Run 4x Super-Resolution
                  </>
                )}
              </button>

              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={handleCompare}
                  disabled={loading || (inputTab === "samples" && !selectedSample) || (inputTab === "upload" && !uploadedFile)}
                  className="py-2.5 px-3 rounded-lg border border-slate-700 bg-slate-900/90 hover:bg-slate-850 hover:border-cyan-500/50 text-xs font-semibold text-cyan-300 transition flex items-center justify-center gap-1.5 shadow-sm"
                >
                  {loading && loadingAction === "compare" ? (
                    <span>Benchmarking...</span>
                  ) : (
                    <>
                      <span>⚖️</span> Compare All Models
                    </>
                  )}
                </button>

                <button
                  onClick={handleAsyncSubmit}
                  disabled={loading || (inputTab === "samples" && !selectedSample) || (inputTab === "upload" && !uploadedFile)}
                  className="py-2.5 px-3 rounded-lg border border-slate-700 bg-slate-900/90 hover:bg-slate-850 hover:border-indigo-500/50 text-xs font-semibold text-indigo-300 transition flex items-center justify-center gap-1.5 shadow-sm"
                >
                  {loading && loadingAction === "async" ? (
                    <span>Submitting...</span>
                  ) : (
                    <>
                      <span>⚡</span> Async Batch Job
                    </>
                  )}
                </button>
              </div>

              {jobStatusMsg && (
                <div className="p-2.5 rounded-lg bg-indigo-950/40 border border-indigo-800/60 text-xs text-indigo-300 font-mono">
                  {jobStatusMsg}
                </div>
              )}
            </div>

            {error && (
              <div className="p-3 rounded-lg bg-rose-950/60 border border-rose-800/80 text-xs text-rose-300">
                ⚠️ {error}
              </div>
            )}
          </div>

          {/* 3. Export & Verification Download Card */}
          {selectedSample && (
            <div className="glass-panel p-4 space-y-3">
              <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                Export Calibrated Imagery & Report
              </h3>
              <p className="text-[11px] text-slate-400">
                Direct export for GIS integration (QGIS / ArcGIS / GDAL) and hackathon validation:
              </p>
              <div className="grid grid-cols-2 gap-2">
                <a
                  href={getGeoTIFFDownloadUrl(selectedSample, selectedModel)}
                  target="_blank"
                  rel="noreferrer"
                  className="py-2 px-3 rounded-lg bg-slate-900 border border-slate-800 hover:border-emerald-500/50 text-xs text-emerald-300 text-center font-medium transition block"
                >
                  📥 4-Band GeoTIFF (.tif)
                </a>
                <a
                  href={getReportDownloadUrl(selectedSample, selectedModel)}
                  target="_blank"
                  rel="noreferrer"
                  className="py-2 px-3 rounded-lg bg-slate-900 border border-slate-800 hover:border-cyan-500/50 text-xs text-cyan-300 text-center font-medium transition block"
                >
                  📊 Physics Report (.json)
                </a>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Visualization & Scientific Metrics (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          {/* Multi-Model Comparison Matrix Card */}
          {compareResult && (
            <div className="glass-panel p-5 space-y-4 border-cyan-500/40 shadow-lg shadow-cyan-950/30">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <span className="text-lg">⚖️</span>
                  <div>
                    <h3 className="text-sm font-semibold text-slate-200">
                      Multi-Model Benchmark Matrix
                    </h3>
                    <p className="text-[11px] text-slate-400">
                      Side-by-side evaluation across standard Bicubic, SRCNN Baseline, and RCAN Attention
                    </p>
                  </div>
                </div>

                {/* Model Tab Switcher for the Viewer below */}
                <div className="flex rounded-lg bg-slate-950 p-1 border border-slate-800 text-xs font-medium">
                  {Object.keys(compareResult.models).map((mId) => (
                    <button
                      key={mId}
                      onClick={() => setActiveCompareModel(mId)}
                      className={`px-3 py-1 rounded transition uppercase text-[11px] ${
                        activeCompareModel === mId
                          ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold"
                          : "text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      {mId}
                    </button>
                  ))}
                </div>
              </div>

              {/* Comparison Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400 font-mono text-[11px]">
                      <th className="py-2 px-3">Metric</th>
                      <th className="py-2 px-3">Bicubic Baseline</th>
                      <th className="py-2 px-3">SRCNN Baseline</th>
                      <th className="py-2 px-3">RCAN Attention</th>
                      <th className="py-2 px-3">Winning Model</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {compareResult.comparison_table.map((row, idx) => (
                      <tr key={idx} className="hover:bg-slate-900/40 transition">
                        <td className="py-2.5 px-3 text-slate-300 font-sans font-medium">
                          {row.metric} {row.unit && <span className="text-slate-500">({row.unit})</span>}
                        </td>
                        <td className="py-2.5 px-3 text-slate-400">
                          {row.bicubic !== undefined && row.bicubic !== null ? row.bicubic : "—"}
                        </td>
                        <td className="py-2.5 px-3 text-slate-300">
                          {row.srcnn !== undefined && row.srcnn !== null ? row.srcnn : "—"}
                        </td>
                        <td className="py-2.5 px-3 text-cyan-300 font-semibold">
                          {row.rcan !== undefined && row.rcan !== null ? row.rcan : "—"}
                        </td>
                        <td className="py-2.5 px-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold ${
                              row.best_model === "rcan"
                                ? "bg-indigo-950 text-indigo-300 border border-indigo-800/60"
                                : row.best_model === "srcnn"
                                ? "bg-cyan-950 text-cyan-300 border border-cyan-800/60"
                                : "bg-slate-800 text-slate-400"
                            }`}
                          >
                            {row.best_model || "—"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeSR && activeInput ? (
            <>
              {/* Image Comparison Card */}
              <div className="glass-panel p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                      <span>Super-Resolution Spatial Viewer</span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/60 uppercase font-mono">
                        {activeSR.model_id}
                      </span>
                    </h3>
                    <p className="text-[11px] text-slate-400 mt-0.5">
                      Input: {activeInput.shape.join("×")} → Enhanced Output: {activeSR.shape.join("×")}
                    </p>
                  </div>

                  <div className="text-right">
                    <span className="text-xs font-mono font-semibold text-cyan-400">
                      {(activeSR.inference_time_s * 1000).toFixed(0)} ms
                    </span>
                    <span className="text-[10px] text-slate-500 block">CPU Latency</span>
                  </div>
                </div>

                {/* Slider / Side-by-side component with Multi-Spectral Channel switcher */}
                <ImageComparisonSlider
                  beforeSrc={activeInput.image}
                  afterSrc={activeSR.image}
                  beforeViews={activeInput.views}
                  afterViews={activeSR.views}
                  groundTruthSrc={activeGT?.image}
                  groundTruthViews={activeGT?.views}
                  uncertaintySrc={activeSR.uncertainty?.image}
                  beforeLabel={`LR Input (${activeInput.shape[1]}×${activeInput.shape[2]})`}
                  afterLabel={`${activeSR.model_id.toUpperCase()} 4x (${activeSR.shape[1]}×${activeSR.shape[2]})`}
                />
              </div>

              {/* Physical & Spectral Evaluation Dashboard */}
              <div className="glass-panel p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-200">
                      Physical & Spectral Metrics
                    </h3>
                    <p className="text-[11px] text-slate-400">
                      Physics-constrained verification of reconstructed reflectance
                    </p>
                  </div>
                  <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/60">
                    Validated
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">PSNR</span>
                    <div className="text-lg font-bold text-cyan-400 font-mono mt-0.5">
                      {activeSR.metrics.psnr?.value !== null && activeSR.metrics.psnr?.value !== undefined
                        ? `${activeSR.metrics.psnr.value.toFixed(2)} dB`
                        : "N/A"}
                    </div>
                    <span className="text-[10px] text-slate-500 block mt-1 leading-tight">
                      Signal-to-Noise Ratio
                    </span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">SSIM</span>
                    <div className="text-lg font-bold text-indigo-400 font-mono mt-0.5">
                      {activeSR.metrics.ssim?.value !== null && activeSR.metrics.ssim?.value !== undefined
                        ? activeSR.metrics.ssim.value.toFixed(3)
                        : "N/A"}
                    </div>
                    <span className="text-[10px] text-slate-500 block mt-1 leading-tight">
                      Structural Fidelity
                    </span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] text-slate-400">SAM (Spectral)</span>
                      <span className="text-[9px] text-emerald-400 font-semibold">&lt;5° Good</span>
                    </div>
                    <div className="text-lg font-bold text-emerald-400 font-mono mt-0.5">
                      {activeSR.metrics.sam?.value !== null && activeSR.metrics.sam?.value !== undefined
                        ? `${activeSR.metrics.sam.value.toFixed(2)}°`
                        : "N/A"}
                    </div>
                    <span className="text-[10px] text-slate-500 block mt-1 leading-tight">
                      Spectral Angle Mapper
                    </span>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Downsample MAE</span>
                    <div className="text-lg font-bold text-amber-400 font-mono mt-0.5">
                      {activeSR.metrics.downsample_consistency?.value !== null && activeSR.metrics.downsample_consistency?.value !== undefined
                        ? activeSR.metrics.downsample_consistency.value.toFixed(4)
                        : "N/A"}
                    </div>
                    <span className="text-[10px] text-slate-500 block mt-1 leading-tight">
                      Degrade Consistency
                    </span>
                  </div>
                </div>
              </div>

              {/* Pixel Spectral Profile & Radiometric Curve Tool */}
              <div className="glass-panel p-5 space-y-4 border-cyan-500/30">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-base">🔬</span>
                    <div>
                      <h3 className="text-sm font-semibold text-slate-200">
                        Multi-Spectral Radiometric Signature Analyzer
                      </h3>
                      <p className="text-[11px] text-slate-400">
                        Pointwise 4-band reflectance signature across Blue (490nm), Green (560nm), Red (665nm), and NIR (842nm)
                      </p>
                    </div>
                  </div>

                  {/* Preset Points */}
                  {selectedSample && (
                    <div className="flex flex-wrap items-center gap-1.5 text-xs">
                      <span className="text-[10px] text-slate-500 uppercase font-mono mr-1">Presets:</span>
                      <button
                        onClick={() => inspectPixel(selectedSample, 100, 100, selectedModel)}
                        className="px-2 py-0.5 rounded bg-slate-900 hover:bg-slate-850 border border-slate-800 text-[11px] text-emerald-300 transition"
                      >
                        🌿 Vegetation
                      </button>
                      <button
                        onClick={() => inspectPixel(selectedSample, 180, 150, selectedModel)}
                        className="px-2 py-0.5 rounded bg-slate-900 hover:bg-slate-850 border border-slate-800 text-[11px] text-sky-300 transition"
                      >
                        🏢 Built-up
                      </button>
                      <button
                        onClick={() => inspectPixel(selectedSample, 50, 120, selectedModel)}
                        className="px-2 py-0.5 rounded bg-slate-900 hover:bg-slate-850 border border-slate-800 text-[11px] text-amber-300 transition"
                      >
                        🛣️ Soil/Route
                      </button>
                    </div>
                  )}
                </div>

                {pixelProfile ? (
                  <div className="space-y-4">
                    {/* Header info */}
                    <div className="flex flex-wrap items-center justify-between gap-2 p-2.5 rounded-lg bg-slate-950 border border-slate-800/80">
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] text-slate-400 font-mono">
                          Coord: ({pixelProfile.hr_coordinates.x}, {pixelProfile.hr_coordinates.y})
                        </span>
                        <span className="text-[10px] px-2 py-0.5 rounded font-semibold bg-cyan-950 text-cyan-300 border border-cyan-800/60">
                          {pixelProfile.surface_classification}
                        </span>
                      </div>

                      <div className="flex items-center gap-3 text-xs font-mono">
                        <span className="text-slate-400">
                          NDVI (LR): <strong className="text-slate-300">{pixelProfile.ndvi.lr.toFixed(3)}</strong>
                        </span>
                        <span className="text-cyan-400">
                          NDVI (BharatSR): <strong className="text-cyan-300">{pixelProfile.ndvi.sr.toFixed(3)}</strong>
                        </span>
                        {pixelProfile.ndvi.hr !== null && (
                          <span className="text-emerald-400">
                            NDVI (GT): <strong className="text-emerald-300">{pixelProfile.ndvi.hr.toFixed(3)}</strong>
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Spectral Band Bars */}
                    <div className="space-y-2.5">
                      {pixelProfile.bands_data.map((b) => (
                        <div key={b.band} className="space-y-1">
                          <div className="flex items-center justify-between text-[11px] font-mono">
                            <span className="text-slate-300 font-medium">
                              {b.band} ({b.name} • {b.wavelength})
                            </span>
                            <div className="flex items-center gap-4 text-[10px]">
                              <span className="text-slate-500">LR 10m: {b.lr_reflectance.toFixed(3)}</span>
                              <span className="text-cyan-300 font-semibold">SR 2.5m: {b.sr_reflectance.toFixed(3)}</span>
                              {b.hr_reflectance !== null && (
                                <span className="text-emerald-400">Ref: {b.hr_reflectance.toFixed(3)}</span>
                              )}
                            </div>
                          </div>

                          {/* Graphical comparison bar */}
                          <div className="w-full h-2 rounded-full bg-slate-900 border border-slate-800 overflow-hidden flex gap-1 p-0.5 relative">
                            <div
                              className="h-full rounded-full bg-cyan-400 transition-all duration-300"
                              style={{ width: `${Math.min(100, Math.max(0, b.sr_reflectance * 100))}%` }}
                              title={`SR: ${b.sr_reflectance}`}
                            />
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800 text-[11px] text-slate-400 flex items-center gap-2">
                      <span className="text-cyan-400 font-bold">ℹ️</span>
                      <span>{pixelProfile.signature_analysis}</span>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-4 text-xs text-slate-500">
                    {loadingPixel ? "Extracting radiometric curve..." : "Select a preset point above to view spectral profile"}
                  </div>
                )}
              </div>

              {/* Uncertainty Quantification Dashboard */}
              {activeSR.uncertainty && (
                <div className="glass-panel p-5 space-y-4 border-amber-500/30">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-amber-400 font-bold text-base">🛡️</span>
                      <div>
                        <h3 className="text-sm font-semibold text-slate-200">
                          Spatial Uncertainty Quantification
                        </h3>
                        <p className="text-[11px] text-slate-400">
                          Per-pixel heteroscedastic variance σ predicted by secondary RCAN network head
                        </p>
                      </div>
                    </div>
                    <span className="text-[10px] font-semibold px-2.5 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800/60">
                      Physics-Calibrated
                    </span>
                  </div>

                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                      <span className="text-[10px] text-slate-400 block">Mean Uncertainty (σ)</span>
                      <div className="text-lg font-bold text-amber-400 font-mono mt-0.5">
                        {activeSR.uncertainty.summary.mean_sigma.toFixed(4)}
                      </div>
                      <span className="text-[10px] text-slate-500 block mt-1 leading-tight">
                        Average model variance
                      </span>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                      <span className="text-[10px] text-slate-400 block">Peak Uncertainty (σ_max)</span>
                      <div className="text-lg font-bold text-rose-400 font-mono mt-0.5">
                        {activeSR.uncertainty.summary.max_sigma.toFixed(3)}
                      </div>
                      <span className="text-[10px] text-slate-500 block mt-1 leading-tight">
                        Maximum edge variance
                      </span>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                      <span className="text-[10px] text-slate-400 block">Base Certainty (σ_min)</span>
                      <div className="text-lg font-bold text-emerald-400 font-mono mt-0.5">
                        {activeSR.uncertainty.summary.min_sigma.toFixed(4)}
                      </div>
                      <span className="text-[10px] text-slate-500 block mt-1 leading-tight">
                        Flat terrain certainty
                      </span>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                      <span className="text-[10px] text-slate-400 block">High-Variance Boundary</span>
                      <div className="text-lg font-bold text-indigo-400 font-mono mt-0.5">
                        {(activeSR.uncertainty.summary.high_uncertainty_fraction * 100).toFixed(1)}%
                      </div>
                      <span className="text-[10px] text-slate-500 block mt-1 leading-tight">
                        Micro-edge pixels
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            /* Placeholder state before processing */
            <div className="glass-panel p-10 text-center space-y-4 border-dashed border-slate-800">
              <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center mx-auto text-cyan-400 shadow-inner">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <div>
                <h3 className="text-base font-semibold text-slate-200">
                  Ready for Super-Resolution Inference
                </h3>
                <p className="text-xs text-slate-400 max-w-md mx-auto mt-1">
                  Select one of the preloaded Sentinel-2 sample tiles on the left and click <strong>&quot;Run 4x Super-Resolution&quot;</strong> or <strong>&quot;Compare All Models&quot;</strong> to view real-time enhancement and spectral accuracy metrics.
                </p>
              </div>
              <div className="flex items-center justify-center gap-3 pt-2">
                <button
                  onClick={handleProcess}
                  disabled={loading || !selectedSample}
                  className="px-4 py-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 text-xs font-semibold hover:bg-cyan-500/20 transition"
                >
                  Quick Demo: Run Sample 0
                </button>
                <button
                  onClick={handleCompare}
                  disabled={loading || !selectedSample}
                  className="px-4 py-2 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 text-xs font-semibold hover:bg-indigo-500/20 transition"
                >
                  Quick Benchmark: Compare 3 Models
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* NTRO SIH26142 Mission Briefing Modal */}
      {showMissionModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-950 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 space-y-5 max-h-[85vh] overflow-y-auto shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <span className="text-xl">🛡️</span>
                <div>
                  <h3 className="text-base font-bold text-slate-100">
                    BharatSR — NTRO Mission Briefing & SIH26142 Spec
                  </h3>
                  <span className="text-xs text-cyan-400 font-mono">
                    Problem Statement SIH26142 • Space Technology
                  </span>
                </div>
              </div>
              <button
                onClick={() => setShowMissionModal(false)}
                className="p-1 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-900 transition"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4 text-xs text-slate-300 leading-relaxed">
              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
                <h4 className="font-semibold text-cyan-300 text-sm">1. Operational Problem Framing</h4>
                <p className="text-slate-400">
                  Medium-resolution Earth observation satellites (Sentinel-2 at 10m–60m, Landsat at 15m–30m) provide continuous strategic coverage of India and its borders, but cannot resolve micro-land-cover boundaries, military vehicle perimeters, or disaster road blockage. High-resolution commercial satellites are cost-prohibitive with narrow coverage swaths. BharatSR provides a physics-constrained 4x spatial super-resolution system (10m $\to$ 2.5m GSD) preserving radiometric integrity.
                </p>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
                <h4 className="font-semibold text-indigo-300 text-sm">2. Key Scientific Differentiators</h4>
                <ul className="list-disc pl-4 space-y-1 text-slate-400">
                  <li><strong>Strict Reflectance Normalization:</strong> Operates directly on calibrated surface reflectance $[0, \sim 1+]$ without ImageNet distortion.</li>
                  <li><strong>Spectral Angle Mapper (SAM &lt; 5°):</strong> Prevents color hallucination and preserves physical band ratios.</li>
                  <li><strong>Downsample Consistency MAE:</strong> Ensures local area degradation matches the original physical sensor input.</li>
                  <li><strong>Spatial Uncertainty Quantification:</strong> Heteroscedastic variance $\sigma$ map flags high-frequency edges to prevent blind AI trust in defense intelligence.</li>
                  <li><strong>Multi-Spectral Analytical Switching:</strong> Instantaneous inspection of True Color RGB, False Color Infrared (CIR), and NDVI vegetation vigor.</li>
                </ul>
              </div>

              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1.5">
                <h4 className="font-semibold text-emerald-300 text-sm">3. Defense GIS Integration</h4>
                <p className="text-slate-400">
                  BharatSR outputs directly to calibrated 4-Band Float32 GeoTIFF (.tif) with geographic affine transforms, fully compatible with standard defense workstations running QGIS, ArcGIS, or GDAL pipelines.
                </p>
              </div>
            </div>

            <div className="pt-2 border-t border-slate-800 flex justify-end">
              <button
                onClick={() => setShowMissionModal(false)}
                className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition"
              >
                Return to Dashboard
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
