"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useConsoleStore } from "@/lib/store";
import {
  Search,
  Sparkles,
  GitCompare,
  BarChart3,
  Sprout,
  Maximize2,
  History,
  Target,
  ListFilter,
  Download,
  Database,
  LayoutDashboard,
  Layers,
  Cpu,
  ArrowRight,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface CommandItem {
  id: string;
  category: "Workspace" | "Model" | "Target Scene";
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  action: () => void;
}

export function CommandPalette() {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const setSelectedModel = useConsoleStore((s) => s.setSelectedModel);
  const setSelectedSample = useConsoleStore((s) => s.setSelectedSample);

  // Global keydown listener for Cmd/Ctrl+K
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setIsOpen((prev) => !prev);
      } else if (e.key === "Escape") {
        setIsOpen(false);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Focus input on open
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  const COMMANDS: CommandItem[] = [
    {
      id: "nav-overview",
      category: "Workspace",
      title: "Mission Console Overview",
      description: "Launch dashboard, guided workflow steps, and system summary",
      icon: LayoutDashboard,
      action: () => router.push("/console"),
    },
    {
      id: "nav-run",
      category: "Workspace",
      title: "Super-Resolution Inference",
      description: "Execute 4× deep learning SR on 10m Sentinel-2 tiles to 2.5m",
      icon: Sparkles,
      action: () => router.push("/console/run"),
    },
    {
      id: "nav-compare",
      category: "Workspace",
      title: "Multi-Model Comparison",
      description: "Side-by-side ablation comparison of SRCNN, RCAN, SwinIR, and HAT",
      icon: GitCompare,
      action: () => router.push("/console/compare"),
    },
    {
      id: "nav-indices",
      category: "Workspace",
      title: "Spectral Indices Analysis",
      description: "Inspect NDVI, NDWI, NDRE, EVI, and SAVI preservation",
      icon: BarChart3,
      action: () => router.push("/console/indices"),
    },
    {
      id: "nav-crop-health",
      category: "Workspace",
      title: "Crop Health & Canopy Assessment",
      description: "Vegetation vigor, LAI, and agricultural canopy stress maps",
      icon: Sprout,
      action: () => router.push("/console/crop-health"),
    },
    {
      id: "nav-field-boundary",
      category: "Workspace",
      title: "Field Boundary Delineation",
      description: "Sub-pixel agricultural plot and parcel edge detection",
      icon: Maximize2,
      action: () => router.push("/console/field-boundary"),
    },
    {
      id: "nav-change-detection",
      category: "Workspace",
      title: "Multi-Temporal Change Detection",
      description: "Bi-temporal surface difference and urban dynamics mapping",
      icon: History,
      action: () => router.push("/console/change-detection"),
    },
    {
      id: "nav-downstream",
      category: "Workspace",
      title: "Downstream Segmentation Tasks",
      description: "Building footprint, road network, and waterbody mask extraction",
      icon: Target,
      action: () => router.push("/console/downstream"),
    },
    {
      id: "nav-jobs",
      category: "Workspace",
      title: "Batch Queue & Job Orchestration",
      description: "Track asynchronous tile inference and worker status",
      icon: ListFilter,
      action: () => router.push("/console/jobs"),
    },
    {
      id: "nav-export",
      category: "Workspace",
      title: "Geospatial Exports & GeoTIFFs",
      description: "Download 32-bit Float32 GeoTIFF rasters and QA compliance certificates",
      icon: Download,
      action: () => router.push("/console/export"),
    },
    {
      id: "nav-dataset",
      category: "Workspace",
      title: "Dataset & Training Transparency",
      description: "Inspect training data provenance, 4-stage preprocessing, and model cards",
      icon: Database,
      action: () => router.push("/console/dataset"),
    },
    // Model switches
    {
      id: "model-rcan",
      category: "Model",
      title: "Switch Model: RCAN-Lite",
      description: "Production baseline with residual channel attention and uncertainty head",
      icon: Cpu,
      action: () => setSelectedModel("rcan"),
    },
    {
      id: "model-hat",
      category: "Model",
      title: "Switch Model: HAT Transformer",
      description: "Hybrid Attention Transformer with local & global self-attention",
      icon: Cpu,
      action: () => setSelectedModel("hat"),
    },
    {
      id: "model-swinir",
      category: "Model",
      title: "Switch Model: SwinIR",
      description: "Swin Transformer for satellite image restoration",
      icon: Cpu,
      action: () => setSelectedModel("swinir"),
    },
    {
      id: "model-srcnn",
      category: "Model",
      title: "Switch Model: SRCNN Baseline",
      description: "Classical 3-layer convolutional baseline",
      icon: Cpu,
      action: () => setSelectedModel("srcnn"),
    },
    // Scene switches
    {
      id: "scene-bellary",
      category: "Target Scene",
      title: "Load Scene: Karnataka Bellary Mining",
      description: "Sentinel-2 L2A tile over iron-ore mining and scrubland",
      icon: Layers,
      action: () => setSelectedSample("sample_real_s2"),
    },
    {
      id: "scene-indore",
      category: "Target Scene",
      title: "Load Scene: MP Indore Kharif",
      description: "Malwa Plateau agricultural soybean corridor",
      icon: Layers,
      action: () => setSelectedSample("scene_cdse_mp_indore_kharif"),
    },
    {
      id: "scene-amritsar",
      category: "Target Scene",
      title: "Load Scene: Punjab Amritsar Kharif",
      description: "Indo-Gangetic agricultural wheat & rice belt",
      icon: Layers,
      action: () => setSelectedSample("scene_cdse_punjab_amritsar_kharif"),
    },
  ];

  const filteredCommands = COMMANDS.filter((cmd) => {
    const q = query.toLowerCase().trim();
    if (!q) return true;
    return (
      cmd.title.toLowerCase().includes(q) ||
      cmd.description.toLowerCase().includes(q) ||
      cmd.category.toLowerCase().includes(q)
    );
  });

  const handleSelect = (cmd: CommandItem) => {
    cmd.action();
    setIsOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % Math.max(1, filteredCommands.length));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev - 1 + filteredCommands.length) % Math.max(1, filteredCommands.length));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (filteredCommands[selectedIndex]) {
        handleSelect(filteredCommands[selectedIndex]);
      }
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20 p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="fixed inset-0" onClick={() => setIsOpen(false)} aria-hidden="true" />

      <div className="relative z-10 w-full max-w-2xl rounded-2xl border border-zinc-800 bg-zinc-950 shadow-2xl shadow-black overflow-hidden flex flex-col max-h-[75vh]">
        {/* Search Input Bar */}
        <div className="flex items-center gap-3 px-4 py-3.5 border-b border-zinc-800 bg-zinc-900/60">
          <Search className="w-5 h-5 text-amber-400 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            onKeyDown={handleKeyDown}
            placeholder="Jump to workspace, switch model, or select scene…"
            className="w-full bg-transparent text-sm text-zinc-100 placeholder:text-zinc-500 font-mono outline-none"
          />
          <kbd className="hidden sm:inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono bg-zinc-800 border border-zinc-700 text-zinc-400">
            ESC
          </kbd>
          <button
            type="button"
            onClick={() => setIsOpen(false)}
            className="p-1 text-zinc-400 hover:text-zinc-200 transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Results List */}
        <div className="flex-1 overflow-y-auto p-2 divide-y divide-zinc-900">
          {filteredCommands.length === 0 ? (
            <div className="p-8 text-center text-xs font-mono text-zinc-500">
              No commands or workspaces found matching &quot;{query}&quot;.
            </div>
          ) : (
            filteredCommands.map((cmd, idx) => {
              const Icon = cmd.icon;
              const isSelected = idx === selectedIndex;
              return (
                <div
                  key={cmd.id}
                  onClick={() => handleSelect(cmd)}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  className={cn(
                    "flex items-center justify-between gap-3 px-3.5 py-2.5 rounded-xl cursor-pointer transition select-none",
                    isSelected
                      ? "bg-amber-500/15 border border-amber-500/40 text-zinc-100"
                      : "text-zinc-400 hover:bg-zinc-900/60 hover:text-zinc-200 border border-transparent"
                  )}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div
                      className={cn(
                        "w-8 h-8 rounded-lg flex items-center justify-center shrink-0 border transition-colors",
                        isSelected
                          ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                          : "bg-zinc-900 text-zinc-400 border-zinc-800"
                      )}
                    >
                      <Icon className="w-4 h-4" />
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold truncate font-sans text-zinc-100">
                          {cmd.title}
                        </span>
                        <span className="px-1.5 py-0.2 rounded font-mono text-[9px] uppercase font-semibold bg-zinc-900 border border-zinc-800 text-zinc-500">
                          {cmd.category}
                        </span>
                      </div>
                      <p className="text-[11px] text-zinc-400 font-sans truncate mt-0.5">
                        {cmd.description}
                      </p>
                    </div>
                  </div>

                  <ArrowRight
                    className={cn(
                      "w-4 h-4 shrink-0 transition-opacity",
                      isSelected ? "text-amber-400 opacity-100" : "opacity-0"
                    )}
                  />
                </div>
              );
            })
          )}
        </div>

        {/* Footer shortcuts */}
        <div className="px-4 py-2 border-t border-zinc-800/80 bg-zinc-950 flex items-center justify-between text-[11px] font-mono text-zinc-500">
          <div className="flex items-center gap-3">
            <span><kbd className="bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-800">↑↓</kbd> navigate</span>
            <span><kbd className="bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-800">↵</kbd> select</span>
          </div>
          <span>Shortcut: <kbd className="bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-800">Ctrl+K</kbd></span>
        </div>
      </div>
    </div>
  );
}
