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

  const connect = useCallback(() => {
    if (!jobId || typeof window === "undefined") return;

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
        if (data.status) setStatus(data.status);
        if (data.result) setResult(data.result);
        if (data.status === "completed" || data.status === "failed") {
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
      // Reconnect with exponential backoff if not completed and attempts < 5
      if (status !== "completed" && status !== "failed" && attemptsRef.current < 5) {
        const delay = Math.min(1000 * 2 ** attemptsRef.current, 10000);
        attemptsRef.current += 1;
        reconnectTimeoutRef.current = setTimeout(connect, delay);
      }
    };
  }, [jobId, status]);

  useEffect(() => {
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
