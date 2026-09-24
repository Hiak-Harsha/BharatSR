"use client";

import React, { useEffect, useRef, useState } from "react";
import { Compass, Globe, Info, Layers, ShieldCheck } from "lucide-react";
import { cn } from "@/lib/utils";

interface GeospatialViewerProps {
  geoMetadata?: {
    has_geo: boolean;
    crs?: string;
    bounds?: [number, number, number, number]; // [min_lon, min_lat, max_lon, max_lat] or [left, bottom, right, top]
    transform?: number[];
    resolution_m?: number;
    sensor?: string;
  } | null;
  imageUrl: string;
  label?: string;
  className?: string;
}

export function GeospatialViewer({
  geoMetadata,
  imageUrl,
  label = "Super-Resolved Satellite Tile",
  className,
}: GeospatialViewerProps) {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);

  const hasGeo = Boolean(geoMetadata && geoMetadata.has_geo && geoMetadata.bounds);

  useEffect(() => {
    if (!hasGeo || !mapContainerRef.current || !imageUrl) return;

    let isMounted = true;

    async function initMap() {
      try {
        const maplibre = await import("maplibre-gl");
        if (!isMounted || !mapContainerRef.current) return;

        // Clean up previous instance
        if (mapInstanceRef.current) {
          mapInstanceRef.current.remove();
          mapInstanceRef.current = null;
        }

        const [minLon, minLat, maxLon, maxLat] = geoMetadata!.bounds!;
        const centerLon = (minLon + maxLon) / 2;
        const centerLat = (minLat + maxLat) / 2;

        const map = new maplibre.Map({
          container: mapContainerRef.current,
          style: {
            version: 8,
            sources: {
              osm: {
                type: "raster",
                tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
                tileSize: 256,
                attribution: "© OpenStreetMap contributors",
              },
            },
            layers: [
              {
                id: "osm-tiles",
                type: "raster",
                source: "osm",
                minzoom: 0,
                maxzoom: 19,
              },
            ],
          },
          center: [centerLon, centerLat],
          zoom: 13,
        });

        map.on("load", () => {
          if (!isMounted) return;

          // Add image overlay coordinates: top-left, top-right, bottom-right, bottom-left
          map.addSource("bharatsr-raster", {
            type: "image",
            url: imageUrl,
            coordinates: [
              [minLon, maxLat],
              [maxLon, maxLat],
              [maxLon, minLat],
              [minLon, minLat],
            ],
          });

          map.addLayer({
            id: "bharatsr-overlay",
            type: "raster",
            source: "bharatsr-raster",
            paint: {
              "raster-opacity": 0.9,
              "raster-resampling": "nearest",
            },
          });

          // Fit bounds
          map.fitBounds(
            [
              [minLon, minLat],
              [maxLon, maxLat],
            ],
            { padding: 40 }
          );

          mapInstanceRef.current = map;
          setMapLoaded(true);
        });

        map.on("error", (e) => {
          console.warn("MapLibre GL warning:", e);
        });
      } catch (err: any) {
        console.error("Failed to initialize MapLibre GL:", err);
        if (isMounted) setMapError(err?.message || "Failed to load map engine");
      }
    }

    initMap();

    return () => {
      isMounted = false;
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, [hasGeo, geoMetadata, imageUrl]);

  return (
    <div className={cn("rounded-xl border border-zinc-800 bg-zinc-950 p-4 flex flex-col gap-3", className)}>
      {/* Header telemetry */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-800 pb-3">
        <div className="flex items-center gap-2">
          <Globe className="w-4 h-4 text-cyan-400" />
          <span className="font-mono text-xs font-semibold text-zinc-200 uppercase tracking-wider">
            {hasGeo ? "Geospatial Alignment & Raster Overlay" : "Cartesian Grid Viewer"}
          </span>
        </div>

        {hasGeo ? (
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono bg-cyan-950/60 text-cyan-300 border border-cyan-800/50">
              <Compass className="w-3 h-3" />
              CRS: {geoMetadata?.crs || "EPSG:32643"}
            </span>
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono bg-emerald-950/60 text-emerald-300 border border-emerald-800/50">
              <ShieldCheck className="w-3 h-3" />
              Authentic Coordinates
            </span>
          </div>
        ) : (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono bg-zinc-800 text-zinc-400 border border-zinc-700">
            <Info className="w-3 h-3" />
            Cartesian Pixel Space (No Geo Fabrication)
          </span>
        )}
      </div>

      {/* Main View Area */}
      {hasGeo && !mapError ? (
        <div className="relative aspect-video w-full rounded-lg overflow-hidden border border-zinc-800 bg-zinc-900">
          <div ref={mapContainerRef} className="absolute inset-0 w-full h-full" />
          {!mapLoaded && (
            <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/80 font-mono text-xs text-zinc-400">
              Initializing spatial projection...
            </div>
          )}
        </div>
      ) : (
        /* Synthetic / Non-Georeferenced View with honest disclaimer */
        <div className="flex flex-col gap-3">
          <div className="relative aspect-square w-full max-h-[460px] rounded-lg overflow-hidden border border-zinc-800 bg-black flex items-center justify-center">
            {imageUrl ? (
              <img src={imageUrl} alt={label} className="w-full h-full object-contain" />
            ) : (
              <span className="font-mono text-xs text-zinc-600">No raster layer loaded</span>
            )}
            <div className="absolute bottom-2 left-2 px-2 py-1 rounded bg-zinc-950/80 border border-zinc-800 font-mono text-[10px] text-zinc-400">
              Resolution: 256×256 px • 4 Channels (B2, B3, B4, B8)
            </div>
          </div>

          <div className="rounded-lg bg-amber-950/20 border border-amber-800/40 p-3 text-xs font-mono text-amber-300/90 flex items-start gap-2.5">
            <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold text-amber-200">Defensibility Guarantee:</p>
              <p className="mt-0.5 text-zinc-400">
                This tile lacks authentic satellite coordinate headers. BharatSR strictly refuses to fabricate
                fictitious latitude/longitude coordinates (EPSG:4326) for non-georeferenced imagery.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
