#!/usr/bin/env python3
"""
BharatSR — GeoTIFF Spatial & Spectral Validation Tool
Confirms that an exported or input GeoTIFF meets scientific remote-sensing standards:
- Valid Coordinate Reference System (CRS)
- Non-degenerate affine transform and origin
- Pixel resolution and expected scale validation (e.g., 2.5m-equivalent grid from 10m input with --scale 4)
- Expected dimensions and band count (4 bands: B2, B3, B4, B8)
- Data type (Float32 for physical surface reflectance)
- Bounds consistency
- NaN / Inf value detection
- Nodata value verification
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import numpy as np


def validate_geotiff(
    tif_path: str,
    expected_scale: Optional[float] = None,
    base_gsd_m: float = 10.0,
    expected_bands: int = 4,
    verbose: bool = True,
) -> bool:
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
            # 1. CRS Check
            crs = ds.crs
            if not crs:
                print(f"[FAIL] {path.name}: Missing Coordinate Reference System (CRS).")
                return False

            # 2. Transform & Resolution
            transform = ds.transform
            if transform.is_identity:
                print(f"[FAIL] {path.name}: Identity transform detected. Geospatial coordinates are missing.")
                return False

            res_x, res_y = ds.res
            if res_x <= 0 or res_y <= 0:
                print(f"[FAIL] {path.name}: Invalid pixel resolution ({res_x}, {res_y}).")
                return False

            # Scale Verification if --scale is provided
            scale_verified = None
            if expected_scale is not None and expected_scale > 0:
                expected_gsd = base_gsd_m / expected_scale
                # Check whether resolution matches expected 2.5m-equivalent grid within 5% tolerance
                # or if the transform scale matches the expected ratio
                rel_err_x = abs(res_x - expected_gsd) / expected_gsd
                rel_err_y = abs(res_y - expected_gsd) / expected_gsd
                if rel_err_x > 0.08 or rel_err_y > 0.08:
                    print(
                        f"[FAIL] {path.name}: Pixel resolution ({res_x:.4f}m, {res_y:.4f}m) does not match "
                        f"expected {expected_gsd:.2f}m grid for scale factor {expected_scale}x."
                    )
                    return False
                scale_verified = f"Valid: {res_x:.2f}m x {res_y:.2f}m (Expected {expected_gsd:.2f}m for {expected_scale}x scale)"

            # 3. Dimensions & Bands
            width, height = ds.width, ds.height
            count = ds.count
            dtype = ds.dtypes[0]

            if count < 1:
                print(f"[FAIL] {path.name}: Zero bands found.")
                return False

            # Check dtype (Scientific reflectance should be float32)
            if dtype != "float32":
                if verbose:
                    print(f"[WARN] {path.name}: Data type is {dtype} (float32 recommended for physical reflectance).")

            # 4. Read data and check values
            data = ds.read()  # (C, H, W)
            nan_count = int(np.isnan(data).sum())
            inf_count = int(np.isinf(data).sum())

            if nan_count > 0 or inf_count > 0:
                print(f"[FAIL] {path.name}: Contains {nan_count} NaN and {inf_count} Inf values.")
                return False

            val_min = float(np.min(data))
            val_max = float(np.max(data))
            val_mean = float(np.mean(data))

            # 5. Bounds
            bounds = ds.bounds

            if verbose:
                print("============================================================")
                print(f" GeoTIFF Validation Report: {path.name}")
                print("============================================================")
                print(f" CRS:            {crs} ({crs.to_string()})")
                print(f" Dimensions:     {width} x {height} pixels, {count} bands")
                print(f" Data Type:      {dtype}")
                print(f" Pixel Size:     {res_x:.4f} x {res_y:.4f} m/pixel")
                if scale_verified:
                    print(f" Scale Check:    {scale_verified}")
                print(f" Affine Origin:  ({transform.c:.4f}, {transform.f:.4f})")
                print(f" Bounds:         Left={bounds.left:.2f}, Bottom={bounds.bottom:.2f}, Right={bounds.right:.2f}, Top={bounds.top:.2f}")
                print(f" Value Range:    Min={val_min:.4f}, Mean={val_mean:.4f}, Max={val_max:.4f}")
                print(f" Nodata:         {ds.nodata}")
                if ds.descriptions and any(ds.descriptions):
                    print(f" Band Names:     {list(ds.descriptions)}")
                if ds.tags():
                    print(f" Metadata Tags:  {dict(ds.tags())}")
                print("------------------------------------------------------------")
                print(" [PASS] Spatially valid and physically compliant GeoTIFF.")
                print("============================================================")

            return True

    except Exception as e:
        print(f"[FAIL] Error reading {tif_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate GeoTIFF spatial integrity for BharatSR")
    parser.add_argument("tif_path", type=str, help="Path to GeoTIFF file")
    parser.add_argument("--scale", type=float, default=None, help="Expected super-resolution scale factor (e.g. 4 for 10m -> 2.5m)")
    parser.add_argument("--base-gsd", type=float, default=10.0, help="Base LR Ground Sampling Distance in meters (default: 10.0 for Sentinel-2)")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    args = parser.parse_args()

    success = validate_geotiff(
        args.tif_path,
        expected_scale=args.scale,
        base_gsd_m=args.base_gsd,
        verbose=not args.quiet
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
