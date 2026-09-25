"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { getWebSocketInferenceUrl } from "@/lib/api-client";

interface WebSocketMessage {
  status: string;
  progress_pct: number;
  message?: string;
  result?: any;
}

export function useJobSocket(jobId: string | null) {
  const [progress, setProgress] = useState<number>(0);
  const [status, setStatus] = useState<string>("idle");
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const attemptsRef = useRef<number>(0);
  const isTerminalRef = useRef<boolean>(false);

  const connect = useCallback(() => {
    if (!jobId || typeof window === "undefined") return;
    if (isTerminalRef.current) return;

    const wsUrl = getWebSocketInferenceUrl(jobId);
    const ws = new WebSocket(wsUrl);
    socketRef.current = ws;

    ws.onopen = () => {
      attemptsRef.current = 0;
      setStatus("connected");
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const data: WebSocketMessage = JSON.parse(event.data);
        if (data.progress_pct !== undefined) setProgress(data.progress_pct);
        if (data.status) {
          setStatus(data.status);
          if (["completed", "failed", "cancelled"].includes(data.status)) {
            isTerminalRef.current = true;
          }
        }
        if (data.result) setResult(data.result);
        if (["completed", "failed", "cancelled"].includes(data.status)) {
          ws.close();
        }
      } catch (err) {
        console.error("Failed to parse websocket message:", err);
      }
    };

    ws.onerror = (err) => {
      console.warn("WebSocket connection error:", err);
      setError("WebSocket link interrupted");
    };

    ws.onclose = () => {
      // Reconnect with exponential backoff ONLY if not in terminal state
      if (!isTerminalRef.current && attemptsRef.current < 5) {
        const delay = Math.min(1000 * 2 ** attemptsRef.current, 10000);
        attemptsRef.current += 1;
        reconnectTimeoutRef.current = setTimeout(connect, delay);
      }
    };
  }, [jobId]);

  useEffect(() => {
    isTerminalRef.current = false;
    attemptsRef.current = 0;
    if (jobId) {
      connect();
    }
    return () => {
      if (socketRef.current) socketRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, [jobId, connect]);

  return { progress, status, result, error };
}
