"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  children?: ReactNode;
  fallbackTitle?: string;
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error?: Error;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Feature panel error caught by boundary:", error, errorInfo);
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: undefined });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="rounded-xl border border-rose-800/40 bg-rose-950/20 p-6 text-zinc-200">
          <div className="flex items-start gap-3">
            <div className="p-2 rounded-lg bg-rose-900/40 text-rose-400 border border-rose-700/50">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div className="flex-1">
              <h3 className="font-mono text-sm font-semibold text-rose-300 uppercase tracking-wider">
                {this.props.fallbackTitle || "Feature Component Encountered An Error"}
              </h3>
              <p className="mt-1 text-xs text-zinc-400 font-mono">
                {this.state.error?.message || "An unexpected error occurred during rendering."}
              </p>
              <div className="mt-4 flex gap-2">
                <button
                  type="button"
                  onClick={this.handleReset}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-medium bg-zinc-800 hover:bg-zinc-700 text-zinc-200 border border-zinc-700 transition"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  Retry Panel
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
