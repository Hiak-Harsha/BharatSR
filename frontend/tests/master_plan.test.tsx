import React from "react";
import { describe, it, expect, beforeEach } from "vitest";
import { useConsoleStore } from "@/lib/store";
import {
  getGeoTIFFDownloadUrl,
  getReportDownloadUrl,
  ApiError,
} from "@/lib/api-client";

describe("BharatSR Master Plan - Canonical Run & Export URLs", () => {
  beforeEach(() => {
    useConsoleStore.setState({
      selectedSample: null,
      currentRunId: null,
      currentSession: null,
      selectedModel: "rcan",
      currentQuality: "fast",
    });
  });

  it("prioritizes run_id over sample_id for GeoTIFF export URL", () => {
    const url = getGeoTIFFDownloadUrl("sample_real_s2", "run_test_12345", "rcan");
    expect(url).toContain("/api/export/geotiff?run_id=run_test_12345&model_id=rcan");
    expect(url).not.toContain("sample_id");
  });

  it("prioritizes run_id over sample_id for Verification Report export URL", () => {
    const url = getReportDownloadUrl("sample_real_s2", "run_test_12345", "rcan");
    expect(url).toContain("/api/export/report?run_id=run_test_12345&model_id=rcan");
    expect(url).not.toContain("sample_id");
  });

  it("falls back to sample_id when run_id is omitted", () => {
    const url = getReportDownloadUrl("sample_real_s2", undefined, "rcan");
    expect(url).toContain("/api/export/report?sample_id=sample_real_s2&model_id=rcan");
  });
});

describe("BharatSR Master Plan - AnalysisSession Store Canonical Concepts", () => {
  beforeEach(() => {
    useConsoleStore.setState({
      selectedSample: null,
      currentRunId: null,
      currentSession: null,
    });
  });

  it("initializes with selectedSample as null (never hardcoded to sample_1)", () => {
    const state = useConsoleStore.getState();
    expect(state.selectedSample).toBeNull();
  });

  it("sets currentRunId and auto-hydrates currentSession", () => {
    const store = useConsoleStore.getState();
    store.setCurrentRunId("run_live_789");

    const updated = useConsoleStore.getState();
    expect(updated.currentRunId).toBe("run_live_789");
    expect(updated.currentSession).not.toBeNull();
    expect(updated.currentSession?.runId).toBe("run_live_789");
  });

  it("fully hydrates session concept with provenance and metadata", () => {
    const store = useConsoleStore.getState();
    store.hydrateSession({
      runId: "run_s2_full",
      sampleId: "sample_real_s2",
      modelId: "rcan",
      quality: "fast",
      status: "completed",
      metrics: { psnr: 34.2, ssim: 0.92, sam: 2.8 },
    });

    const updated = useConsoleStore.getState();
    expect(updated.currentRunId).toBe("run_s2_full");
    expect(updated.currentSession?.metrics?.psnr).toBe(34.2);
    expect(updated.currentSession?.status).toBe("completed");
  });
});

describe("BharatSR Master Plan - API Error Normalization Layer", () => {
  it("creates an ApiError with actionToFix, endpoint, and technicalDetails", () => {
    const error = new ApiError({
      status: 400,
      message: "No geospatial reference available",
      actionToFix: "Upload a georeferenced raster.",
      endpoint: "/api/export/geotiff",
      technicalDetails: "Status: 400 | Endpoint: /api/export/geotiff | Request ID: req_123",
      requestId: "req_123",
    });

    expect(error.status).toBe(400);
    expect(error.detail).toContain("No geospatial reference available");
    expect(error.actionToFix).toContain("Upload a georeferenced raster");
    expect(error.requestId).toBe("req_123");
    expect(error.endpoint).toBe("/api/export/geotiff");
  });
});
