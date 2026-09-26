import type { Metadata } from "next";
import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";
import { QueryProvider } from "@/components/providers/QueryProvider";
import { BootSplash } from "@/components/effects/BootSplash";

export const metadata: Metadata = {
  title: "BharatSR — Physics-Constrained Satellite Super-Resolution",
  description:
    "Physics-constrained deep learning super-resolution mapping for medium-resolution satellite imagery — SIH26142 (NTRO)",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-[var(--bsr-bg)] text-[var(--bsr-ink)] min-h-screen antialiased">
        <QueryProvider>
          <BootSplash>{children}</BootSplash>
        </QueryProvider>
      </body>
    </html>
  );
}
