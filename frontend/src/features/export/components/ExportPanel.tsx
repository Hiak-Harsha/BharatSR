"use client";

import React from "react";
import { useConsoleStore } from "@/lib/store";
import { getGeoTIFFDownloadUrl, getReportDownloadUrl } from "@/lib/api-client";
import { Download, FileText, Globe, ShieldAlert, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

export function ExportPanel({ className }: { className?: string }) {
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const selectedModel = useConsoleStore((s) => s.selectedModel);
  const currentRunId = useConsoleStore((s) => s.currentRunId);

  const geotiffUrl = getGeoTIFFDownloadUrl(
    selectedSample || undefined,
    currentRunId || undefined,
    selectedModel
  );

  const reportUrl = getReportDownloadUrl(
    selectedSample || undefined,
    currentRunId || undefined,
    selectedModel
  );

  return (
    <div className={cn("flex flex-col gap-6", className)}>
      <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950">
        <h2 className="font-mono text-base font-bold text-zinc-100 flex items-center gap-2">
          <Download className="w-5 h-5 text-amber-400" />
          Authoritative Geospatial & Analytical Exports
        </h2>
        <p className="mt-1 font-mono text-xs text-zinc-400">
          Download 32-bit floating point GeoTIFF imagery and SIH verification reports
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* GeoTIFF Export Card */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-6 flex flex-col justify-between gap-4">
          <div className="space-y-3 font-mono">
            <div className="w-10 h-10 rounded-lg bg-cyan-950/80 border border-cyan-800/60 flex items-center justify-center text-cyan-300">
              <Globe className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-zinc-100 uppercase tracking-wider">
              Authoritative 4-Band Float32 GeoTIFF
            </h3>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Export 4-channel super-resolved raster (B2 Blue, B3 Green, B4 Red, B8 NIR).
              Preserves authentic Coordinate Reference System (CRS) and precisely scales affine transform for 4× resolution.
            </p>

            <div className="rounded-lg bg-amber-950/20 border border-amber-800/40 p-2.5 text-[11px] text-amber-300/80 flex items-start gap-2">
              <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <span>
                Integrity Rule: If the dataset lacks geospatial headers, export halts with HTTP 400.
                EPSG coordinates are never fabricated.
              </span>
            </div>
          </div>

          <a
            href={geotiffUrl}
            download
            className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg font-mono text-xs font-bold uppercase bg-amber-500 hover:bg-amber-400 text-zinc-950 transition shadow-lg shadow-amber-500/10 text-center"
          >
            <Download className="w-4 h-4" />
            Download 4-Band GeoTIFF (.tif)
          </a>
        </div>

        {/* Verification Report Card */}
        <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-6 flex flex-col justify-between gap-4">
          <div className="space-y-3 font-mono">
            <div className="w-10 h-10 rounded-lg bg-amber-950/80 border border-amber-800/60 flex items-center justify-center text-amber-300">
              <FileText className="w-5 h-5" />
            </div>
            <h3 className="text-sm font-bold text-zinc-100 uppercase tracking-wider">
              SIH Verification & Compliance Report
            </h3>
            <p className="text-xs text-zinc-400 leading-relaxed">
              Comprehensive analytical summary in JSON format detailing PSNR, SSIM, SAM,
              downsample consistency, latency benchmarks, and defensibility declarations.
            </p>

            <div className="rounded-lg bg-zinc-900 border border-zinc-800 p-2.5 text-[11px] text-zinc-400 flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <span>
                Suitable for automated testing pipelines and technical audit verification.
              </span>
            </div>
          </div>

          <a
            href={reportUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg font-mono text-xs font-bold uppercase bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700 transition text-center"
          >
            <FileText className="w-4 h-4" />
            View SIH Verification JSON
          </a>
        </div>
      </div>
    </div>
  );
}
