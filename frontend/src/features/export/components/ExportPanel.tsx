"use client";

import React from "react";
import { useConsoleStore } from "@/lib/store";
import { getGeoTIFFDownloadUrl, getReportDownloadUrl } from "@/lib/api-client";
import {
  Download,
  FileText,
  Globe,
  ShieldCheck,
  CheckCircle2,
  HardDrive,
  ExternalLink,
  Layers,
  ArrowRight,
} from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";

export function ExportPanel({ className }: { className?: string }) {
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const selectedModel = useConsoleStore((s) => s.selectedModel);
  const currentRunId = useConsoleStore((s) => s.currentRunId);
  const currentSession = useConsoleStore((s) => s.currentSession);
  const activeModel = currentSession?.modelId || selectedModel;
  const activeScene = currentSession?.sampleId || selectedSample || "sample_real_s2";

  const geotiffUrl = getGeoTIFFDownloadUrl(
    currentRunId ? undefined : (selectedSample || undefined),
    currentRunId || undefined,
    activeModel
  );

  const reportUrl = getReportDownloadUrl(
    currentRunId ? undefined : (selectedSample || undefined),
    currentRunId || undefined,
    activeModel
  );

  const exportItems = [
    {
      id: "geotiff",
      name: "4-Band Float32 Georeferenced Surface Reflectance",
      filename: `${currentRunId || activeScene}_${activeModel}_2.5m.tif`,
      format: "GeoTIFF (.tif)",
      type: "Raster / Surface BOA",
      estimatedSize: "~2.2 MB",
      runId: currentRunId || "Sample Session",
      description:
        "Multi-band georeferenced raster with scaled affine transform (p_out = p_in / 4). Compatible with QGIS, ArcGIS, GDAL, and Google Earth Engine.",
      downloadUrl: geotiffUrl,
      icon: Globe,
      isPrimary: true,
    },
    {
      id: "report",
      name: "Scientific Verification Certificate & Fidelity Metrics",
      filename: `${currentRunId || activeScene}_verification_report.json`,
      format: "JSON (.json)",
      type: "Audit / QA Report",
      estimatedSize: "~14 KB",
      runId: currentRunId || "Sample Session",
      description:
        "Machine-readable evaluation manifest containing SAM angle, downsample consistency MAE, PSNR, SSIM, and defensibility declarations.",
      downloadUrl: reportUrl,
      icon: FileText,
      isPrimary: false,
    },
  ];

  return (
    <div className={cn("space-y-6", className)}>
      {/* Session Context Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 rounded-xl border border-zinc-800 bg-zinc-900/50">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wider text-amber-400 font-semibold">
            ACTIVE EXPORT CONTEXT
          </div>
          <h2 className="text-lg font-bold text-zinc-100 mt-0.5">
            Production Delivery Packages
          </h2>
          <p className="text-xs text-zinc-400 font-sans mt-0.5">
            Download georeferenced products generated with zero data alteration and strict geospatial provenance.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs font-mono">
          <div className="px-3 py-1.5 rounded-lg bg-zinc-950 border border-zinc-800">
            <span className="text-zinc-500">Run ID:</span>{" "}
            <strong className="text-amber-400">
              {currentRunId || "Pre-loaded Canonical Run"}
            </strong>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-zinc-950 border border-zinc-800">
            <span className="text-zinc-500">Model:</span>{" "}
            <strong className="text-zinc-200 uppercase">{activeModel}</strong>
          </div>
          <div className="px-3 py-1.5 rounded-lg bg-zinc-950 border border-zinc-800">
            <span className="text-zinc-500">Grid:</span>{" "}
            <strong className="text-emerald-400">2.5m Ground Resolution</strong>
          </div>
        </div>
      </div>

      {/* Exportable Files Table */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-950 overflow-hidden shadow-xl">
        <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-amber-400" />
            <h3 className="font-mono text-sm font-bold text-zinc-200 uppercase tracking-wider">
              Exportable Product Manifest ({exportItems.length} Available)
            </h3>
          </div>
          <span className="text-[11px] font-mono text-zinc-500">
            Direct HTTP Download • Integrity Verified
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead className="bg-zinc-900/60 border-b border-zinc-800 text-zinc-400 uppercase text-[10px] tracking-wider">
              <tr>
                <th className="py-3 px-4">Product Name &amp; Description</th>
                <th className="py-3 px-4">Format</th>
                <th className="py-3 px-4">Size</th>
                <th className="py-3 px-4">Run Association</th>
                <th className="py-3 px-4 text-right">Download Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-850">
              {exportItems.map((item) => {
                const Icon = item.icon;
                return (
                  <tr key={item.id} className="hover:bg-zinc-900/40 transition">
                    <td className="py-4 px-4 max-w-md">
                      <div className="flex items-start gap-3">
                        <div className="w-8 h-8 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center text-amber-400 shrink-0 mt-0.5">
                          <Icon className="w-4 h-4" />
                        </div>
                        <div>
                          <div className="font-bold text-zinc-100 text-sm font-sans">
                            {item.name}
                          </div>
                          <div className="text-[11px] text-zinc-400 font-sans mt-0.5 leading-relaxed">
                            {item.description}
                          </div>
                          <div className="text-[10px] text-zinc-500 font-mono mt-1">
                            File: {item.filename}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="py-4 px-4 whitespace-nowrap">
                      <span className="px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300 font-semibold text-[11px]">
                        {item.format}
                      </span>
                    </td>
                    <td className="py-4 px-4 text-zinc-400 whitespace-nowrap">
                      {item.estimatedSize}
                    </td>
                    <td className="py-4 px-4 whitespace-nowrap">
                      <span className="text-amber-400 font-bold text-[11px]">
                        {item.runId.slice(0, 16)}
                      </span>
                    </td>
                    <td className="py-4 px-4 text-right whitespace-nowrap">
                      <a
                        href={item.downloadUrl}
                        download
                        className={cn(
                          "inline-flex items-center gap-1.5 px-4 py-2 rounded-lg font-mono text-xs font-bold transition shadow-sm",
                          item.isPrimary
                            ? "bg-amber-500 hover:bg-amber-400 text-zinc-950 shadow-amber-500/20"
                            : "bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700"
                        )}
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>Download</span>
                      </a>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Geospatial Governance & Compliance Card */}
      <div className="p-4 rounded-xl border border-zinc-800 bg-zinc-950 font-mono text-xs space-y-2">
        <div className="flex items-center gap-2 text-zinc-300 font-bold uppercase tracking-wider text-[11px]">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          Strict Geospatial Integrity Guarantee
        </div>
        <p className="text-zinc-400 text-xs font-sans leading-relaxed">
          Exported rasters retain the exact affine projection matrix scaled by 4× spatial resolution, preserving original Sentinel-2 geotransforms without synthetic coordinate fabrication. All exported files conform to standard GDAL, rasterio, and OGC geospatial raster requirements.
        </p>
      </div>
    </div>
  );
}
