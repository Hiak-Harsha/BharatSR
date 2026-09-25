import { describe, it, expect, beforeEach } from "vitest";
import { useConsoleStore } from "@/lib/store";

describe("useConsoleStore", () => {
  beforeEach(() => {
    useConsoleStore.setState({
      selectedModel: "rcan",
      selectedSample: null,
      customFile: null,
      activeView: "single",
      activeBandView: "rgb",
      comparisonSliderPos: 50,
      currentRunId: null,
      currentQuality: "fast",
      currentSession: null,
    });
  });

  it("initializes with default model and null sample", () => {
    const state = useConsoleStore.getState();
    expect(state.selectedModel).toBe("rcan");
    expect(state.selectedSample).toBeNull();
    expect(state.currentRunId).toBeNull();
    expect(state.currentSession).toBeNull();
  });

  it("updates selected model and resets run state", () => {
    useConsoleStore.getState().setSelectedModel("srcnn");
    expect(useConsoleStore.getState().selectedModel).toBe("srcnn");
  });

  it("sets custom file and unsets selectedSample", () => {
    const file = new File(["dummy"], "sentinel2.tif", { type: "image/tiff" });
    useConsoleStore.getState().setCustomFile(file);

    expect(useConsoleStore.getState().customFile).toBe(file);
    expect(useConsoleStore.getState().selectedSample).toBeNull();
  });

  it("propagates currentRunId correctly", () => {
    useConsoleStore.getState().setCurrentRunId("run_test12345");
    expect(useConsoleStore.getState().currentRunId).toBe("run_test12345");

    useConsoleStore.getState().resetRun();
    expect(useConsoleStore.getState().currentRunId).toBeNull();
  });

  it("switches active console views and bands", () => {
    useConsoleStore.getState().setActiveView("compare");
    expect(useConsoleStore.getState().activeView).toBe("compare");

    useConsoleStore.getState().setActiveBandView("cir");
    expect(useConsoleStore.getState().activeBandView).toBe("cir");
  });
});
