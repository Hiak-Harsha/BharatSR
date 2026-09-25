"use client";

import React from "react";
import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export interface WorkspaceShellProps {
  title: string;
  description: string;
  icon: LucideIcon;
  status?: React.ReactNode;
  sidebar?: React.ReactNode;
  children: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}

export function WorkspaceShell({
  title,
  description,
  icon: Icon,
  status,
  sidebar,
  children,
  actions,
  className,
}: WorkspaceShellProps) {
  return (
    <div className={cn("w-full max-w-[1600px] mx-auto flex flex-col gap-6", className)}>
      {/* Workspace Header Frame */}
      <div className="rounded-2xl border border-zinc-800/80 bg-zinc-950/80 backdrop-blur-md p-6 sm:p-7 shadow-lg shadow-black/40 transition-all">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="flex items-start sm:items-center gap-4">
            <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400 shrink-0 shadow-[0_0_15px_rgba(245,158,11,0.15)]">
              <Icon className="w-6 h-6" />
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-3">
                <h1 className="text-2xl font-bold tracking-tight text-zinc-100">
                  {title}
                </h1>
                {status && <div className="inline-flex items-center">{status}</div>}
              </div>
              <p className="text-sm text-zinc-400 mt-1 font-sans leading-relaxed max-w-2xl">
                {description}
              </p>
            </div>
          </div>

          {actions && (
            <div className="flex items-center gap-3 shrink-0 self-start sm:self-auto">
              {actions}
            </div>
          )}
        </div>
      </div>

      {/* Main Workspace Body */}
      {sidebar ? (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Per-workspace Sidebar (e.g. Model & Sample Selectors for Run & Compare) */}
          <aside className="lg:col-span-4 xl:col-span-3 flex flex-col gap-6 w-full">
            {sidebar}
          </aside>

          {/* Primary Content Area */}
          <main className="lg:col-span-8 xl:col-span-9 w-full min-w-0">
            <div className="rounded-2xl border border-zinc-800/80 bg-zinc-950/80 backdrop-blur-md p-6 sm:p-8 shadow-lg shadow-black/40 min-h-[500px]">
              {children}
            </div>
          </main>
        </div>
      ) : (
        /* Full-Width Workspace Canvas (No Sidebar) */
        <main className="w-full">
          <div className="rounded-2xl border border-zinc-800/80 bg-zinc-950/80 backdrop-blur-md p-6 sm:p-8 shadow-lg shadow-black/40 min-h-[500px]">
            {children}
          </div>
        </main>
      )}
    </div>
  );
}
