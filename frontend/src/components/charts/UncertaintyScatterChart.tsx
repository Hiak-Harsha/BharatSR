"use client";

import React from "react";
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { UncertaintySummary } from "@/lib/api-client";
import { cn } from "@/lib/utils";

interface UncertaintyScatterChartProps {
  scatter?: {
    correlation: number;
    points: Array<{ unc: number; err: number }>;
  };
  summary?: UncertaintySummary;
  className?: string;
}

export function UncertaintyScatterChart({
  scatter,
  summary,
  className,
}: UncertaintyScatterChartProps) {
  if (!scatter || !scatter.points || scatter.points.length === 0) {
    return (
      <div className={cn("h-48 flex items-center justify-center font-mono text-xs text-zinc-500", className)}>
        Uncertainty calibration scatter requires ground-truth comparison data.
      </div>
    );
  }

  const data = scatter.points.map((pt, i) => ({
    x: Number(pt.unc.toFixed(4)),
    y: Number(pt.err.toFixed(4)),
    index: i,
  }));

  return (
    <div className={cn("flex flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-950 p-4", className)}>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-850 pb-2">
        <h4 className="font-mono text-xs font-semibold text-zinc-200 uppercase tracking-wider">
          Uncertainty Calibration (σ vs |SR - GT|)
        </h4>
        <span className="px-2 py-0.5 rounded text-xs font-mono bg-cyan-950/60 text-cyan-300 border border-cyan-800/60">
          Pearson r: {scatter.correlation.toFixed(3)}
        </span>
      </div>

      <div className="h-48 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis
              type="number"
              dataKey="x"
              name="Predicted Uncertainty (σ)"
              stroke="#71717a"
              tick={{ fontSize: 10, fontFamily: "monospace" }}
              domain={[0, "auto"]}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Actual Error (|SR - GT|)"
              stroke="#71717a"
              tick={{ fontSize: 10, fontFamily: "monospace" }}
              domain={[0, "auto"]}
            />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              contentStyle={{
                backgroundColor: "#09090b",
                borderColor: "#27272a",
                borderRadius: "8px",
                fontFamily: "monospace",
                fontSize: "11px",
              }}
            />
            <Scatter name="Validation Points" data={data} fill="#38bdf8" fillOpacity={0.6} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {summary && (
        <div className="grid grid-cols-3 gap-2 pt-2 border-t border-zinc-850 text-xs font-mono">
          <div className="bg-zinc-900/60 p-2 rounded">
            <span className="text-zinc-500 block text-[10px]">Mean σ</span>
            <span className="text-zinc-200 font-bold">
              {(Number((summary as any).mean_sigma ?? (summary as any).mean_uncertainty ?? 0)).toFixed(4)}
            </span>
          </div>
          <div className="bg-zinc-900/60 p-2 rounded">
            <span className="text-zinc-500 block text-[10px]">Max σ</span>
            <span className="text-zinc-200 font-bold">
              {(Number((summary as any).max_sigma ?? (summary as any).max_uncertainty ?? 0)).toFixed(4)}
            </span>
          </div>
          <div className="bg-zinc-900/60 p-2 rounded">
            <span className="text-zinc-500 block text-[10px]">High Unc. Ratio</span>
            <span className="text-amber-300 font-bold">
              {(summary as any).high_uncertainty_fraction != null
                ? `${(Number((summary as any).high_uncertainty_fraction) * 100).toFixed(1)}%`
                : (summary as any).high_uncertainty_pixel_pct != null
                ? `${Number((summary as any).high_uncertainty_pixel_pct).toFixed(1)}%`
                : "N/A"}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
