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
      <div className={cn("h-64 flex items-center justify-center font-mono text-sm text-zinc-400", className)}>
        Sampling multi-spectral reflectance across B2, B3, B4, B8...
      </div>
    );
  }

  if (!data || !data.bands_data || data.bands_data.length === 0) {
    return (
      <div className={cn("h-64 flex flex-col items-center justify-center font-mono text-sm text-zinc-400 border border-dashed border-zinc-800 rounded-lg p-6 text-center", className)}>
        <span>No pixel selected. Click anywhere on the satellite canvas to inspect spectral signature and reflectance values.</span>
      </div>
    );
  }

  const chartData = data.bands_data.map((b) => ({
    name: `${b.band} (${b.name})`,
    wavelength: b.wavelength,
    LR: Number((b.lr_reflectance ?? 0).toFixed(4)),
    Bicubic: b.bicubic_reflectance != null ? Number(b.bicubic_reflectance.toFixed(4)) : undefined,
    BharatSR: Number((b.sr_reflectance ?? 0).toFixed(4)),
    Reference: b.hr_reflectance != null ? Number(b.hr_reflectance.toFixed(4)) : undefined,
  }));

  const hasReference = chartData.some((d) => d.Reference !== undefined);

  return (
    <div className={cn("flex flex-col gap-4 rounded-xl border border-zinc-800 bg-zinc-950 p-4", className)}>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-850 pb-3">
        <div>
          <h4 className="font-mono text-sm font-semibold text-zinc-100 uppercase tracking-wider">
            Multi-Spectral Pixel Profile
          </h4>
          <div className="font-mono text-xs text-zinc-400 mt-0.5 flex items-center gap-3">
            <span>SR Grid: <strong className="text-zinc-200">[{data.hr_coordinates.x}, {data.hr_coordinates.y}]</strong></span>
            <span>·</span>
            <span>Corresponding LR: <strong className="text-zinc-200">[{data.lr_coordinates.x}, {data.lr_coordinates.y}]</strong></span>
          </div>
        </div>
        {data.surface_classification && (
          <span className="px-2.5 py-1 rounded text-xs font-mono bg-amber-500/10 text-amber-300 border border-amber-500/30">
            {data.surface_classification}
          </span>
        )}
      </div>

      {/* Chart */}
      <div className="h-60 w-full">
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
            <Legend wrapperStyle={{ fontSize: "12px", fontFamily: "monospace", paddingTop: "8px" }} />
            <Line type="monotone" dataKey="LR" stroke="#94a3b8" strokeWidth={2} strokeDasharray="4 4" dot={{ r: 3 }} />
            {chartData[0]?.Bicubic !== undefined && (
              <Line type="monotone" dataKey="Bicubic" stroke="#f59e0b" strokeWidth={1.5} dot={{ r: 3 }} />
            )}
            <Line type="monotone" dataKey="BharatSR" stroke="#38bdf8" strokeWidth={2.5} dot={{ r: 4 }} />
            {hasReference && (
              <Line type="monotone" dataKey="Reference" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Detailed Values Breakdown Table */}
      <div className="overflow-x-auto rounded-lg border border-zinc-850">
        <table className="w-full text-left font-mono text-xs">
          <thead className="bg-zinc-900/80 text-zinc-400 border-b border-zinc-850">
            <tr>
              <th className="py-2 px-3">Band / Index</th>
              <th className="py-2 px-3">LR (10m)</th>
              <th className="py-2 px-3">Bicubic (4x)</th>
              <th className="py-2 px-3 text-cyan-300">BharatSR (2.5m)</th>
              {hasReference && <th className="py-2 px-3 text-emerald-300">Reference</th>}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-850/60 bg-zinc-950/40">
            {data.bands_data.map((b) => (
              <tr key={b.band}>
                <td className="py-2 px-3 font-semibold text-zinc-300">
                  {b.band} ({b.name})
                </td>
                <td className="py-2 px-3 text-zinc-400">{(b.lr_reflectance ?? 0).toFixed(4)}</td>
                <td className="py-2 px-3 text-zinc-400">{b.bicubic_reflectance != null ? b.bicubic_reflectance.toFixed(4) : "—"}</td>
                <td className="py-2 px-3 font-bold text-cyan-400">{(b.sr_reflectance ?? 0).toFixed(4)}</td>
                {hasReference && (
                  <td className="py-2 px-3 text-emerald-400">{b.hr_reflectance != null ? b.hr_reflectance.toFixed(4) : "—"}</td>
                )}
              </tr>
            ))}
            <tr className="bg-zinc-900/40 font-semibold border-t border-zinc-800">
              <td className="py-2 px-3 text-amber-300">NDVI</td>
              <td className="py-2 px-3 text-zinc-400">{data.ndvi.lr?.toFixed(4) ?? "—"}</td>
              <td className="py-2 px-3 text-zinc-400">{data.ndvi.bicubic != null ? (data.ndvi.bicubic as number).toFixed(4) : "—"}</td>
              <td className="py-2 px-3 font-bold text-cyan-300">{data.ndvi.sr?.toFixed(4) ?? "—"}</td>
              {hasReference && (
                <td className="py-2 px-3 text-emerald-300">{data.ndvi.hr?.toFixed(4) ?? "—"}</td>
              )}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
