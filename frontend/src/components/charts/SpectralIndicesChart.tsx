"use client";

import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { cn } from "@/lib/utils";

interface SpectralIndicesChartProps {
  indices: Record<string, {
    sr: { mean: number; std?: number; p25?: number; p75?: number };
    lr: { mean: number; std?: number };
  }>;
  className?: string;
}

export function SpectralIndicesChart({ indices, className }: SpectralIndicesChartProps) {
  if (!indices || Object.keys(indices).length === 0) return null;

  const data = Object.entries(indices).map(([name, stat]) => ({
    name: name.toUpperCase(),
    "LR Mean": Number((stat.lr?.mean ?? 0).toFixed(3)),
    "SR Mean": Number((stat.sr?.mean ?? 0).toFixed(3)),
    "Uplift (Δ)": Number(((stat.sr?.mean ?? 0) - (stat.lr?.mean ?? 0)).toFixed(3)),
  }));

  return (
    <div className={cn("flex flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-950 p-4", className)}>
      <div className="flex items-center justify-between border-b border-zinc-850 pb-2">
        <h4 className="font-mono text-xs font-semibold text-zinc-200 uppercase tracking-wider">
          Spectral Index Fidelity (LR vs BharatSR)
        </h4>
        <span className="text-[11px] font-mono text-zinc-500">NDVI • EVI • NDWI • SAVI • NDRE</span>
      </div>

      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis dataKey="name" stroke="#71717a" tick={{ fontSize: 11, fontFamily: "monospace" }} />
            <YAxis stroke="#71717a" tick={{ fontSize: 11, fontFamily: "monospace" }} />
            <Tooltip
              contentStyle={{
                backgroundColor: "#09090b",
                borderColor: "#27272a",
                borderRadius: "8px",
                fontFamily: "monospace",
                fontSize: "12px",
              }}
            />
            <Legend wrapperStyle={{ fontSize: "11px", fontFamily: "monospace" }} />
            <Bar dataKey="LR Mean" fill="#64748b" radius={[4, 4, 0, 0]} />
            <Bar dataKey="SR Mean" fill="#38bdf8" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
