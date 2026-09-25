"use client";

import React, { useEffect } from "react";
import { X, Layers } from "lucide-react";
import { ModelSelector } from "@/features/models/components/ModelSelector";
import { SampleGallery } from "@/features/samples/components/SampleGallery";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";

interface SampleModelPickerDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SampleModelPickerDrawer({ isOpen, onClose }: SampleModelPickerDrawerProps) {
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        onClose();
      }
    }
    if (isOpen) {
      window.addEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "hidden";
    }
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "unset";
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/75 backdrop-blur-sm transition-opacity animate-in fade-in duration-200">
      {/* Backdrop click to dismiss */}
      <div className="fixed inset-0" onClick={onClose} aria-hidden="true" />

      {/* Drawer content */}
      <div className="relative z-10 w-full max-w-xl h-full bg-zinc-950 border-l border-zinc-800 shadow-2xl flex flex-col animate-in slide-in-from-right duration-250">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800/80 bg-zinc-900/50">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400">
              <Layers className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-bold text-zinc-100">Select Scene &amp; Model</h2>
              <p className="text-xs text-zinc-400 font-mono">Global Console Configuration</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-2 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/60 transition"
            aria-label="Close Drawer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Scrollable selectors */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          <ErrorBoundary fallbackTitle="Model Selection Error">
            <ModelSelector />
          </ErrorBoundary>

          <ErrorBoundary fallbackTitle="Sample Gallery Error">
            <SampleGallery />
          </ErrorBoundary>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-zinc-800/80 bg-zinc-900/40 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-5 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-zinc-950 font-semibold text-xs transition shadow-md shadow-amber-500/20"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
