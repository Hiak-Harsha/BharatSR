import { create } from "zustand";

export type ConsoleViewMode =
  | "single"
  | "compare"
  | "indices"
  | "crop_health"
  | "field_boundary"
  | "change_detection"
  | "downstream"
  | "jobs"
  | "export";

export type BandViewMode =
  | "composite"
  | "rgb"
  | "cir"
  | "ndvi"
  | "red"
  | "green"
  | "blue"
  | "nir"
  | "error";

export interface AnalysisSession {
  runId: string;
  jobId?: string | null;
  sampleId?: string | null;
  fileName?: string | null;
  inputType: "sample" | "upload";
  modelId: string;
  quality: "fast" | "high";
  status: "completed" | "processing" | "failed";
  createdAt: string;
  inputMetadata?: Record<string, any> | null;
  outputMetadata?: Record<string, any> | null;
  metrics?: Record<string, any> | null;
  uncertainty?: Record<string, any> | null;
  geospatialMetadata?: Record<string, any> | null;
  provenance?: string | null;
}

interface ConsoleUIState {
  selectedModel: string;
  selectedSample: string | null;
  customFile: File | null;
  activeView: ConsoleViewMode;
  activeBandView: BandViewMode;
  comparisonSliderPos: number;
  currentRunId: string | null;
  currentQuality: "fast" | "high";
  currentSession: AnalysisSession | null;
  
  // Actions
  setSelectedModel: (modelId: string) => void;
  setSelectedSample: (sampleId: string | null) => void;
  setCustomFile: (file: File | null) => void;
  setActiveView: (view: ConsoleViewMode) => void;
  setActiveBandView: (band: BandViewMode) => void;
  setComparisonSliderPos: (pos: number) => void;
  setCurrentRunId: (runId: string | null) => void;
  setCurrentQuality: (quality: "fast" | "high") => void;
  setCurrentSession: (session: AnalysisSession | null) => void;
  hydrateSession: (session: Partial<AnalysisSession> & { runId: string }) => void;
  resetRun: () => void;
}

export const useConsoleStore = create<ConsoleUIState>((set) => ({
  selectedModel: "rcan",
  selectedSample: null,
  customFile: null,
  activeView: "single",
  activeBandView: "composite",
  comparisonSliderPos: 50,
  currentRunId: null,
  currentQuality: "fast",
  currentSession: null,

  setSelectedModel: (selectedModel) => set({ selectedModel }),
  setSelectedSample: (selectedSample) => set({ selectedSample, customFile: null }),
  setCustomFile: (customFile) => set({ customFile, selectedSample: null }),
  setActiveView: (activeView) => set({ activeView }),
  setActiveBandView: (activeBandView) => set({ activeBandView }),
  setComparisonSliderPos: (comparisonSliderPos) => set({ comparisonSliderPos }),
  setCurrentRunId: (currentRunId) => set((state) => ({
    currentRunId,
    currentSession: state.currentSession && state.currentSession.runId === currentRunId
      ? state.currentSession
      : currentRunId
        ? {
            runId: currentRunId,
            inputType: state.customFile ? "upload" : "sample",
            sampleId: state.selectedSample,
            fileName: state.customFile?.name || null,
            modelId: state.selectedModel,
            quality: state.currentQuality,
            status: "completed",
            createdAt: new Date().toISOString(),
          }
        : null,
  })),
  setCurrentQuality: (currentQuality) => set({ currentQuality }),
  setCurrentSession: (currentSession) => set({
    currentSession,
    currentRunId: currentSession ? currentSession.runId : null,
  }),
  hydrateSession: (session) => set((state) => {
    const fullSession: AnalysisSession = {
      runId: session.runId,
      jobId: session.jobId ?? state.currentSession?.jobId ?? null,
      sampleId: session.sampleId ?? state.selectedSample ?? null,
      fileName: session.fileName ?? state.customFile?.name ?? null,
      inputType: session.inputType ?? (state.customFile ? "upload" : "sample"),
      modelId: session.modelId ?? state.selectedModel,
      quality: session.quality ?? state.currentQuality,
      status: session.status ?? "completed",
      createdAt: session.createdAt ?? state.currentSession?.createdAt ?? new Date().toISOString(),
      inputMetadata: session.inputMetadata ?? state.currentSession?.inputMetadata ?? null,
      outputMetadata: session.outputMetadata ?? state.currentSession?.outputMetadata ?? null,
      metrics: session.metrics ?? state.currentSession?.metrics ?? null,
      uncertainty: session.uncertainty ?? state.currentSession?.uncertainty ?? null,
      geospatialMetadata: session.geospatialMetadata ?? state.currentSession?.geospatialMetadata ?? null,
      provenance: session.provenance ?? state.currentSession?.provenance ?? null,
    };
    return {
      currentRunId: session.runId,
      currentSession: fullSession,
    };
  }),
  resetRun: () => set({ currentRunId: null, currentSession: null }),
}));
