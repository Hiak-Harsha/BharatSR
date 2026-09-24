"use client";

import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { PixelProfileResponse } from "@/lib/api-client";
import { cn } from "@/lib/utils";

interface PixelProfileChartProps {
  data: PixelProfileResponse | null;
  isLoading?: boolean;
  className?: string;
}

export function PixelProfileChart({ data, isLoading, className }: PixelProfileChartProps) {
  if (isLoading) {
    return (
      <div className={cn("h-64 flex items-center justify-center font-mono text-xs text-zinc-500", className)}>
        Sampling multi-spectral reflectance across B2, B3, B4, B8...
      </div>
    );
  }

  if (!data || !data.bands_data || data.bands_data.length === 0) {
    return (
      <div className={cn("h-64 flex flex-col items-center justify-center font-mono text-xs text-zinc-500 border border-dashed border-zinc-800 rounded-lg p-6 text-center", className)}>
        <span>No pixel selected. Click anywhere on the satellite tile canvas to inspect its spectral signature.</span>
      </div>
    );
  }

  const chartData = data.bands_data.map((b) => ({
    name: `${b.band} (${b.name})`,
    wavelength: b.wavelength,
    LR: Number((b.lr_reflectance ?? 0).toFixed(4)),
    Bicubic: b.bicubic_reflectance != null ? Number(b.bicubic_reflectance.toFixed(4)) : undefined,
    BharatSR: Number((b.sr_reflectance ?? 0).toFixed(4)),
    "Ground Truth": b.hr_reflectance != null ? Number(b.hr_reflectance.toFixed(4)) : undefined,
  }));

  return (
    <div className={cn("flex flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-950 p-4", className)}>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-850 pb-2.5">
        <div>
          <h4 className="font-mono text-xs font-semibold text-zinc-200 uppercase tracking-wider">
            Spectral Reflectance Curve
          </h4>
          <span className="font-mono text-[11px] text-zinc-500">
            Coordinates: LR [{data.lr_coordinates.x}, {data.lr_coordinates.y}] • SR [{data.hr_coordinates.x}, {data.hr_coordinates.y}]
          </span>
        </div>
        {data.surface_classification && (
          <span className="px-2 py-0.5 rounded text-xs font-mono bg-amber-500/10 text-amber-300 border border-amber-500/30">
            {data.surface_classification}
          </span>
        )}
      </div>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis dataKey="name" stroke="#71717a" tick={{ fontSize: 11, fontFamily: "monospace" }} />
            <YAxis
              stroke="#71717a"
              tick={{ fontSize: 11, fontFamily: "monospace" }}
              domain={[0, "auto"]}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#09090b",
                borderColor: "#27272a",
                borderRadius: "8px",
                fontFamily: "monospace",
                fontSize: "12px",
              }}
            />
            <Legend wrapperStyle={{ fontSize: "11px", fontFamily: "monospace", paddingTop: "8px" }} />
            <Line type="monotone" dataKey="LR" stroke="#94a3b8" strokeWidth={2} strokeDasharray="4 4" dot={{ r: 3 }} />
            {chartData[0]?.Bicubic !== undefined && (
              <Line type="monotone" dataKey="Bicubic" stroke="#f59e0b" strokeWidth={1.5} dot={{ r: 3 }} />
            )}
            <Line type="monotone" dataKey="BharatSR" stroke="#38bdf8" strokeWidth={2.5} dot={{ r: 4 }} />
            {chartData[0]?.["Ground Truth"] !== undefined && (
              <Line type="monotone" dataKey="Ground Truth" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* NDVI & Spectral Angle Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 pt-2 border-t border-zinc-850 text-xs font-mono">
        <div className="bg-zinc-900/60 p-2 rounded border border-zinc-800">
          <span className="text-zinc-500 block text-[10px]">NDVI (SR)</span>
          <span className="text-cyan-300 font-bold">{data.ndvi.sr?.toFixed(3) ?? "N/A"}</span>
        </div>
        <div className="bg-zinc-900/60 p-2 rounded border border-zinc-800">
          <span className="text-zinc-500 block text-[10px]">NDVI (LR)</span>
          <span className="text-zinc-300 font-bold">{data.ndvi.lr?.toFixed(3) ?? "N/A"}</span>
        </div>
        <div className="bg-zinc-900/60 p-2 rounded border border-zinc-800">
          <span className="text-zinc-500 block text-[10px]">SAM Angle</span>
          <span className="text-amber-300 font-bold">
            {data.spectral_angle_deg != null ? `${data.spectral_angle_deg.toFixed(2)}°` : "N/A"}
          </span>
        </div>
        <div className="bg-zinc-900/60 p-2 rounded border border-zinc-800">
          <span className="text-zinc-500 block text-[10px]">NDVI (GT)</span>
          <span className="text-emerald-300 font-bold">{data.ndvi.hr?.toFixed(3) ?? "N/A"}</span>
        </div>
      </div>
    </div>
  );
}
