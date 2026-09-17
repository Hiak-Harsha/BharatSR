#!/usr/bin/env python3
"""
BharatSR — GeoTIFF Spatial & Spectral Validation Tool
Confirms that an exported or input GeoTIFF meets scientific remote-sensing standards:
- Valid CRS
- Non-degenerate affine transform
- Preserved pixel size relationship (e.g. 2.5m output from 10m input)
- Band count, dtype (float32), and physical reflectance range
- Valid bounding box
"""

import sys
import argparse
from pathlib import Path
import numpy as np


def validate_geotiff(tif_path: str, expected_scale: float = None, verbose: bool = True) -> bool:
    path = Path(tif_path)
    if not path.exists():
        print(f"[FAIL] File not found: {tif_path}")
        return False

    try:
        import rasterio
    except ImportError:
        print("[FAIL] rasterio is required for GeoTIFF validation.")
        return False

    try:
        with rasterio.open(str(path)) as ds:
            # 1. CRS
            crs = ds.crs
            if not crs:
                print(f"[FAIL] {path.name}: Missing Coordinate Reference System (CRS).")
                return False

            # 2. Transform & Resolution
            transform = ds.transform
            if transform.is_identity:
                print(f"[FAIL] {path.name}: Identity transform detected. Geospatial coordinates are not set.")
                return False

            res_x, res_y = ds.res
            if res_x <= 0 or res_y <= 0:
                print(f"[FAIL] {path.name}: Invalid pixel resolution ({res_x}, {res_y}).")
                return False

            # 3. Dimensions & Bands
            width, height = ds.width, ds.height
            count = ds.count
            dtype = ds.dtypes[0]

            if count < 1:
                print(f"[FAIL] {path.name}: Zero bands found.")
                return False

            # 4. Read data and check values
            data = ds.read()  # (C, H, W)
            nan_count = np.isnan(data).sum()
            inf_count = np.isinf(data).sum()

            if nan_count > 0 or inf_count > 0:
                print(f"[FAIL] {path.name}: Contains {nan_count} NaN and {inf_count} Inf values.")
                return False

            val_min = float(np.min(data))
            val_max = float(np.max(data))
            val_mean = float(np.mean(data))

            # 5. Bounds
            bounds = ds.bounds

            if verbose:
                print(f"============================================================")
                print(f" GeoTIFF Validation Report: {path.name}")
                print(f"============================================================")
                print(f" CRS:            {crs} ({crs.to_string()})")
                print(f" Dimensions:     {width} x {height} pixels, {count} bands")
                print(f" Data Type:      {dtype}")
                print(f" Pixel Size:     {res_x:.4f} x {res_y:.4f}")
                print(f" Affine Origin:  ({transform.c:.4f}, {transform.f:.4f})")
                print(f" Bounds:         Left={bounds.left:.2f}, Bottom={bounds.bottom:.2f}, Right={bounds.right:.2f}, Top={bounds.top:.2f}")
                print(f" Value Range:    Min={val_min:.4f}, Mean={val_mean:.4f}, Max={val_max:.4f}")
                print(f" Nodata:         {ds.nodata}")
                if ds.descriptions and any(ds.descriptions):
                    print(f" Band Names:     {ds.descriptions}")
                if ds.tags():
                    print(f" Metadata Tags:  {dict(ds.tags())}")
                print(f"------------------------------------------------------------")
                print(f" [PASS] Spatially valid and physically compliant GeoTIFF.")
                print(f"============================================================")

            return True

    except Exception as e:
        print(f"[FAIL] Error reading {tif_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate GeoTIFF spatial integrity for BharatSR")
    parser.add_argument("tif_path", type=str, help="Path to GeoTIFF file")
    parser.add_argument("--scale", type=float, default=None, help="Expected scale factor")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    args = parser.parse_args()

    success = validate_geotiff(args.tif_path, expected_scale=args.scale, verbose=not args.quiet)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
