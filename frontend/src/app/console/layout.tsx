"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { fetchHealth } from "@/lib/api-client";
import { useConsoleStore } from "@/lib/store";
import { ApiKeyModal } from "@/components/ui/ApiKeyModal";
import { StatusBadge } from "@/components/ui/StatusBadge";
import { SampleModelPickerDrawer } from "@/components/console/SampleModelPickerDrawer";
import { CommandPalette } from "@/components/console/CommandPalette";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Sparkles,
  GitCompare,
  BarChart3,
  Sprout,
  Maximize2,
  History,
  Target,
  ListFilter,
  Download,
  Database,
  Cpu,
  ChevronLeft,
  ChevronRight,
  Menu,
  X,
  Layers,
  Satellite,
  ArrowUpRight,
  Search,
} from "lucide-react";

interface NavItem {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
  showPulse?: boolean;
}

interface NavGroup {
  groupName?: string;
  items: NavItem[];
}

const NAV_GROUPS: NavGroup[] = [
  {
    items: [
      { href: "/console", label: "Overview", icon: LayoutDashboard },
    ],
  },
  {
    groupName: "Resolve",
    items: [
      { href: "/console/run", label: "Super-Resolution", icon: Sparkles },
      { href: "/console/compare", label: "Model Comparison", icon: GitCompare },
    ],
  },
  {
    groupName: "Analyze",
    items: [
      { href: "/console/indices", label: "Spectral Indices", icon: BarChart3 },
      { href: "/console/crop-health", label: "Crop Health", icon: Sprout },
      { href: "/console/field-boundary", label: "Field Boundary", icon: Maximize2 },
      { href: "/console/change-detection", label: "Change Detection", icon: History },
      { href: "/console/downstream", label: "Downstream Tasks", icon: Target },
    ],
  },
  {
    groupName: "Operate",
    items: [
      { href: "/console/jobs", label: "Batch Queue", icon: ListFilter },
      { href: "/console/export", label: "Exports", icon: Download },
    ],
  },
  {
    groupName: "Model & Data",
    items: [
      { href: "/console/dataset", label: "Dataset & Training", icon: Database },
    ],
  },
];

export default function ConsoleLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const selectedModel = useConsoleStore((s) => s.selectedModel);
  const selectedSample = useConsoleStore((s) => s.selectedSample);
  const currentRunId = useConsoleStore((s) => s.currentRunId);

  // Sidebar collapse persistence in localStorage
  const [collapsed, setCollapsed] = useState(false);
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
  const [isPickerOpen, setIsPickerOpen] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("bharatsr_sidebar_collapsed");
      if (saved !== null) {
        setCollapsed(saved === "true");
      }
    }
  }, []);

  const toggleCollapsed = () => {
    const next = !collapsed;
    setCollapsed(next);
    if (typeof window !== "undefined") {
      localStorage.setItem("bharatsr_sidebar_collapsed", String(next));
    }
  };

  // Close mobile drawer on route change
  useEffect(() => {
    setMobileDrawerOpen(false);
  }, [pathname]);

  // Backend telemetry
  const { data: health, isLoading: isHealthLoading, isError: isHealthError } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 10000,
  });

  return (
    <div className="bg-[#090c10] text-zinc-100 min-h-screen flex flex-col antialiased selection:bg-amber-500/20 selection:text-amber-200">
      {/* Top Persistent Status Bar */}
      <header className="border-b border-zinc-800/80 bg-zinc-950/90 backdrop-blur-md sticky top-0 z-40 h-16 flex items-center px-4 sm:px-6 justify-between gap-4">
        {/* Left: Brand & Mobile/Collapse Toggles */}
        <div className="flex items-center gap-3 shrink-0">
          {/* Mobile drawer toggle (<1024px) */}
          <button
            type="button"
            onClick={() => setMobileDrawerOpen(true)}
            className="lg:hidden p-2 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/60 transition"
            aria-label="Open navigation menu"
          >
            <Menu className="w-5 h-5" />
          </button>

          {/* Desktop collapse toggle (>=1024px) */}
          <button
            type="button"
            onClick={toggleCollapsed}
            className="hidden lg:flex p-1.5 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60 transition"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>

          {/* Logo & Console Title */}
          <Link href="/console" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-amber-500 to-indigo-600 flex items-center justify-center shadow-md shadow-amber-500/20 group-hover:scale-105 transition-transform">
              <Satellite className="w-4 h-4 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base tracking-tight text-white">
                  BharatSR
                </span>
                <span className="text-[9px] uppercase font-mono font-semibold px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30">
                  Mission Console
                </span>
              </div>
            </div>
          </Link>
        </div>

        {/* Center: Search / Command Palette Shortcut + Target Context Strip */}
        <div className="hidden md:flex items-center gap-3">
          <button
            type="button"
            onClick={() => {
              window.dispatchEvent(new KeyboardEvent("keydown", { key: "k", ctrlKey: true }));
            }}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-zinc-200 text-xs font-mono transition"
            title="Search or jump to workspace (Ctrl+K)"
          >
            <Search className="w-3.5 h-3.5 text-amber-400" />
            <span>Search workspaces…</span>
            <kbd className="text-[10px] bg-zinc-800 px-1.5 py-0.5 rounded border border-zinc-700 text-zinc-400">Ctrl+K</kbd>
          </button>

          <div className="flex items-center gap-2 bg-zinc-900/80 border border-zinc-800 rounded-lg px-3 py-1 text-xs font-mono">
            <span className="text-zinc-500 uppercase text-[10px] tracking-wider">Target:</span>
            <span className="text-zinc-200 font-medium truncate max-w-[130px]" title={selectedSample || "No sample selected"}>
              {selectedSample || "sample_real_s2"}
            </span>
            <span className="text-zinc-600">•</span>
            <span className="text-zinc-500 uppercase text-[10px] tracking-wider">Model:</span>
            <span className="text-amber-400 font-bold uppercase">
              {selectedModel}
            </span>
            <button
              type="button"
              onClick={() => setIsPickerOpen(true)}
              className="ml-1.5 px-2 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-[10px] font-semibold border border-zinc-700/60 hover:text-amber-300 transition flex items-center gap-1"
            >
              <Layers className="w-3 h-3" />
              Change
            </button>
          </div>
        </div>

        {/* Right: Telemetry & Actions */}
        <div className="flex items-center gap-3 shrink-0">
          {/* Active run ID indicator */}
          {currentRunId && (
            <div className="hidden xl:flex items-center gap-1.5 px-2.5 py-1 rounded bg-zinc-900 border border-amber-500/30 font-mono text-[11px]">
              <span className="text-zinc-500">Active:</span>
              <span className="text-amber-400 font-bold max-w-[100px] truncate" title={currentRunId}>
                {currentRunId}
              </span>
            </div>
          )}

          {/* Device and backend health */}
          <div className="hidden sm:flex items-center gap-2 font-mono text-xs pl-2 border-l border-zinc-800">
            <div className="flex items-center gap-1 text-zinc-400 text-[11px]">
              <Cpu className="w-3 h-3 text-amber-400" />
              <span className="text-zinc-200 uppercase font-semibold">
                {health?.device || "CPU"}
              </span>
            </div>
            {isHealthLoading ? (
              <span className="text-zinc-500 text-[11px]">Checking…</span>
            ) : isHealthError ? (
              <StatusBadge status="Offline" variant="error" />
            ) : health?.degraded ? (
              <StatusBadge status="Degraded" variant="warning" />
            ) : (
              <StatusBadge status={health?.status || "Ready"} variant="success" />
            )}
          </div>

          <ApiKeyModal />

          <Link
            href="/"
            className="hidden sm:flex items-center gap-1 px-2.5 py-1 rounded-lg bg-zinc-900 border border-zinc-800 text-xs text-zinc-300 hover:border-zinc-700 hover:text-white transition"
          >
            <span>Landing</span>
            <ArrowUpRight className="w-3 h-3 text-zinc-500" />
          </Link>
        </div>
      </header>

      {/* Main Layout Container (Sidebar + Content Outlet) */}
      <div className="flex-1 flex w-full min-h-[calc(100vh-4rem)] relative">
        {/* Left Rail Navigation: Desktop (Persistent & Collapsible) */}
        <aside
          className={cn(
            "hidden lg:flex flex-col border-r border-zinc-800/80 bg-zinc-950/70 backdrop-blur-md sticky top-16 h-[calc(100vh-4rem)] transition-all duration-200 z-30 shrink-0 select-none overflow-y-auto overflow-x-hidden",
            collapsed ? "w-18 p-2" : "w-64 p-4"
          )}
        >
          <div className="space-y-6 flex-1">
            {NAV_GROUPS.map((group, gIdx) => (
              <div key={gIdx} className="space-y-1">
                {group.groupName && !collapsed && (
                  <div className="px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider text-zinc-500 font-semibold">
                    {group.groupName}
                  </div>
                )}
                {group.items.map((item) => {
                  const Icon = item.icon;
                  // Exact match for /console, prefix match for sub-routes
                  const isActive =
                    item.href === "/console"
                      ? pathname === "/console"
                      : pathname === item.href || pathname.startsWith(`${item.href}/`);

                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      title={collapsed ? item.label : undefined}
                      className={cn(
                        "flex items-center gap-3 px-3 py-2 rounded-xl font-mono text-xs font-medium transition-all group relative",
                        isActive
                          ? "bg-amber-500/15 text-amber-300 border border-amber-500/40 shadow-[0_0_12px_rgba(245,158,11,0.15)] font-semibold"
                          : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900/80 border border-transparent"
                      )}
                    >
                      <Icon
                        className={cn(
                          "w-4 h-4 shrink-0 transition-colors",
                          isActive ? "text-amber-400" : "text-zinc-500 group-hover:text-zinc-300"
                        )}
                      />
                      {!collapsed && (
                        <span className="truncate font-sans text-xs tracking-tight">
                          {item.label}
                        </span>
                      )}
                      {/* Active indicator bar */}
                      {isActive && (
                        <span className="absolute left-0 top-1.5 bottom-1.5 w-1 rounded-r bg-amber-400 shadow-[0_0_8px_rgba(245,158,11,0.8)]" />
                      )}
                    </Link>
                  );
                })}
              </div>
            ))}
          </div>

          {/* Quick Scene Picker Trigger at bottom of sidebar when expanded */}
          {!collapsed && (
            <div className="pt-4 border-t border-zinc-800/80 mt-4">
              <button
                type="button"
                onClick={() => setIsPickerOpen(true)}
                className="w-full flex items-center justify-between px-3 py-2 rounded-xl bg-zinc-900/60 hover:bg-zinc-850 border border-zinc-800 hover:border-zinc-700 text-xs font-mono text-zinc-300 transition group"
              >
                <div className="flex items-center gap-2 truncate">
                  <Layers className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                  <span className="truncate text-zinc-400 group-hover:text-zinc-200 text-[11px]">
                    Target Scene
                  </span>
                </div>
                <span className="text-[10px] text-amber-400 font-bold uppercase">
                  Select
                </span>
              </button>
            </div>
          )}
        </aside>

        {/* Mobile Slide-Over Navigation Drawer (<1024px) */}
        {mobileDrawerOpen && (
          <div className="fixed inset-0 z-50 lg:hidden flex">
            <div
              className="fixed inset-0 bg-black/80 backdrop-blur-sm transition-opacity"
              onClick={() => setMobileDrawerOpen(false)}
              aria-hidden="true"
            />

            <div className="relative z-10 w-72 max-w-[80vw] h-full bg-zinc-950 border-r border-zinc-800 p-5 flex flex-col overflow-y-auto animate-in slide-in-from-left duration-250">
              <div className="flex items-center justify-between pb-4 border-b border-zinc-800 mb-4">
                <div className="flex items-center gap-2">
                  <Satellite className="w-5 h-5 text-amber-400" />
                  <span className="font-bold text-sm text-white">Mission Console</span>
                </div>
                <button
                  type="button"
                  onClick={() => setMobileDrawerOpen(false)}
                  className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800"
                  aria-label="Close Menu"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-5 flex-1">
                {NAV_GROUPS.map((group, gIdx) => (
                  <div key={gIdx} className="space-y-1">
                    {group.groupName && (
                      <div className="px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider text-zinc-500 font-semibold">
                        {group.groupName}
                      </div>
                    )}
                    {group.items.map((item) => {
                      const Icon = item.icon;
                      const isActive =
                        item.href === "/console"
                          ? pathname === "/console"
                          : pathname === item.href || pathname.startsWith(`${item.href}/`);

                      return (
                        <Link
                          key={item.href}
                          href={item.href}
                          className={cn(
                            "flex items-center gap-3 px-3 py-2.5 rounded-xl font-mono text-xs font-medium transition-all",
                            isActive
                              ? "bg-amber-500/15 text-amber-300 border border-amber-500/40 font-semibold"
                              : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 border border-transparent"
                          )}
                        >
                          <Icon className={cn("w-4 h-4", isActive ? "text-amber-400" : "text-zinc-500")} />
                          <span className="font-sans text-xs">{item.label}</span>
                        </Link>
                      );
                    })}
                  </div>
                ))}
              </div>

              <div className="pt-4 border-t border-zinc-800/80 mt-4 space-y-3">
                <button
                  type="button"
                  onClick={() => {
                    setMobileDrawerOpen(false);
                    setIsPickerOpen(true);
                  }}
                  className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs font-mono font-medium"
                >
                  <Layers className="w-4 h-4" />
                  Change Scene / Model
                </button>
                <Link
                  href="/"
                  className="w-full flex items-center justify-center gap-1.5 py-2 px-3 rounded-xl bg-zinc-900 border border-zinc-800 text-zinc-300 text-xs font-mono"
                >
                  Back to Landing Page
                </Link>
              </div>
            </div>
          </div>
        )}

        {/* Primary Content Outlet */}
        <div className="flex-1 w-full min-w-0 flex flex-col p-4 sm:p-6 lg:p-8">
          {children}
        </div>
      </div>

      {/* Global Sample & Model Picker Drawer */}
      <SampleModelPickerDrawer
        isOpen={isPickerOpen}
        onClose={() => setIsPickerOpen(false)}
      />

      {/* Global Command Palette (Cmd/Ctrl+K) */}
      <CommandPalette />
    </div>
  );
}
