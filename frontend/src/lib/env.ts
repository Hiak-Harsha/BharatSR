import { z } from "zod";

const envSchema = z.object({
  NEXT_PUBLIC_API_URL: z.string().optional().default(""),
  NEXT_PUBLIC_WS_URL: z.string().optional().default(""),
});

const parsed = envSchema.safeParse({
  NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL,
});

if (!parsed.success) {
  console.error("Invalid environment variables:", parsed.error.format());
  throw new Error("Invalid frontend environment configuration.");
}

export const env = parsed.data;

/**
 * Resolves the API base URL based on runtime context.
 * In the browser, an empty string ("") is preferred so Next.js rewrites proxy to backend
 * without triggering cross-origin issues or port mismatches.
 */
export function getApiBase(): string {
  if (env.NEXT_PUBLIC_API_URL && env.NEXT_PUBLIC_API_URL.trim() !== "") {
    return env.NEXT_PUBLIC_API_URL.replace(/\/$/, "");
  }
  return typeof window !== "undefined" ? "" : "http://127.0.0.1:8000";
}

/**
 * Resolves WebSocket URL for real-time inference telemetry.
 */
export function getWsBase(): string {
  if (env.NEXT_PUBLIC_WS_URL && env.NEXT_PUBLIC_WS_URL.trim() !== "") {
    return env.NEXT_PUBLIC_WS_URL.replace(/\/$/, "");
  }
  if (typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}`;
  }
  return "ws://127.0.0.1:8000";
}
