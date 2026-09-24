import React from "react";
import { cn } from "@/lib/utils";

interface StatusBadgeProps {
  status: string;
  variant?: "default" | "success" | "warning" | "error" | "info" | "amber";
  className?: string;
}

export function StatusBadge({ status, variant = "default", className }: StatusBadgeProps) {
  const variantStyles = {
    default: "bg-zinc-800 text-zinc-300 border-zinc-700",
    success: "bg-emerald-950/80 text-emerald-300 border-emerald-700/60",
    warning: "bg-amber-950/80 text-amber-300 border-amber-700/60",
    amber: "bg-amber-950/80 text-amber-400 border-amber-600/60 shadow-[0_0_8px_rgba(245,158,11,0.2)]",
    error: "bg-rose-950/80 text-rose-300 border-rose-700/60",
    info: "bg-cyan-950/80 text-cyan-300 border-cyan-700/60",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono font-medium tracking-wide uppercase border",
        variantStyles[variant],
        className
      )}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
      {status}
    </span>
  );
}
