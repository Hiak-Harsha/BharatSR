"use client";

import React, { useState, useEffect } from "react";
import { Key, Check, X } from "lucide-react";

export function ApiKeyModal() {
  const [isOpen, setIsOpen] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      setApiKey(localStorage.getItem("bharatsr_api_key") || "");
    }
  }, [isOpen]);

  const handleSave = () => {
    if (typeof window !== "undefined") {
      if (apiKey.trim()) {
        localStorage.setItem("bharatsr_api_key", apiKey.trim());
      } else {
        localStorage.removeItem("bharatsr_api_key");
      }
      setSaved(true);
      setTimeout(() => {
        setSaved(false);
        setIsOpen(false);
      }, 800);
    }
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        aria-label="API Key Settings"
        className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-zinc-900 border border-zinc-800 text-xs text-zinc-400 hover:text-amber-400 hover:border-zinc-700 transition"
      >
        <Key className="w-3.5 h-3.5" />
        <span className="hidden md:inline font-mono text-[11px]">
          {apiKey ? "Key: Configured" : "Auth"}
        </span>
      </button>

      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-xs p-4">
          <div className="w-full max-w-md rounded-xl border border-zinc-800 bg-zinc-950 p-5 shadow-2xl font-mono text-xs">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-zinc-100 font-bold text-sm">
                <Key className="w-4 h-4 text-amber-400" />
                API Authorization Settings
              </div>
              <button
                type="button"
                onClick={() => setIsOpen(false)}
                className="text-zinc-500 hover:text-zinc-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <p className="text-zinc-400 text-xs mb-3 leading-relaxed">
              The public BharatSR demo runs in open key-free mode for all inference, comparison, and analysis features. An API key is only needed for operator-restricted admin tasks (e.g. model reload).
            </p>

            <div className="space-y-1.5 mb-4">
              <label className="text-[10px] text-zinc-500 uppercase tracking-wider block">
                Backend Secret Key
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Leave blank for demo / key-free mode"
                className="w-full bg-zinc-900 border border-zinc-800 px-3 py-2 rounded-lg text-zinc-200 focus:border-amber-500 focus:outline-none text-xs"
              />
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-zinc-900">
              <span className="text-[11px] text-zinc-500">
                Stored in browser localStorage
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setIsOpen(false)}
                  className="px-3 py-1.5 rounded-lg border border-zinc-800 text-zinc-400 hover:text-zinc-200"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSave}
                  className="px-4 py-1.5 rounded-lg bg-amber-500 hover:bg-amber-400 text-zinc-950 font-bold flex items-center gap-1.5 transition"
                >
                  {saved ? (
                    <>
                      <Check className="w-3.5 h-3.5" />
                      Saved
                    </>
                  ) : (
                    "Save Key"
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
