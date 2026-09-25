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

interface ConsoleUIState {
  selectedModel: string;
  selectedSample: string | null;
  customFile: File | null;
  activeView: ConsoleViewMode;
  activeBandView: BandViewMode;
  comparisonSliderPos: number;
  currentRunId: string | null;
  currentQuality: "fast" | "high";
  
  // Actions
  setSelectedModel: (modelId: string) => void;
  setSelectedSample: (sampleId: string | null) => void;
  setCustomFile: (file: File | null) => void;
  setActiveView: (view: ConsoleViewMode) => void;
  setActiveBandView: (band: BandViewMode) => void;
  setComparisonSliderPos: (pos: number) => void;
  setCurrentRunId: (runId: string | null) => void;
  setCurrentQuality: (quality: "fast" | "high") => void;
  resetRun: () => void;
}

export const useConsoleStore = create<ConsoleUIState>((set) => ({
  selectedModel: "rcan",
  selectedSample: "sample_1",
  customFile: null,
  activeView: "single",
  activeBandView: "composite",
  comparisonSliderPos: 50,
  currentRunId: null,
  currentQuality: "fast",

  setSelectedModel: (selectedModel) => set({ selectedModel }),
  setSelectedSample: (selectedSample) => set({ selectedSample, customFile: null }),
  setCustomFile: (customFile) => set({ customFile, selectedSample: null }),
  setActiveView: (activeView) => set({ activeView }),
  setActiveBandView: (activeBandView) => set({ activeBandView }),
  setComparisonSliderPos: (comparisonSliderPos) => set({ comparisonSliderPos }),
  setCurrentRunId: (currentRunId) => set({ currentRunId }),
  setCurrentQuality: (currentQuality) => set({ currentQuality }),
  resetRun: () => set({ currentRunId: null }),
}));
