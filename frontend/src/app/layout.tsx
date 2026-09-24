import type { Metadata } from "next";
import "./globals.css";
import { QueryProvider } from "@/components/providers/QueryProvider";

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
        <QueryProvider>{children}</QueryProvider>
      </body>
    </html>
  );
}
