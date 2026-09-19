#!/usr/bin/env python3
"""
Automated GeoTIFF Round-Trip Verification Script for BharatSR (SIH26142).

Verifies the complete end-to-end geospatial lifecycle:
1. Ingests genuine Sentinel-2 Level-2A GeoTIFF (10m GSD, 4 bands, EPSG:32643).
2. Executes 4x super-resolution mapping to 2.5m-equivalent grid.
3. Exports calibrated 4-band Float32 GeoTIFF.
4. Reads exported GeoTIFF with Rasterio and verifies:
   - Resolution: 10m / 4 = 2.5m
   - Dimensions: W_in * 4, H_in * 4 (e.g. 64x64 -> 256x256)
   - Bounding Box: Identical geographic coverage (origin & extent preserved)
   - CRS: Identical Coordinate Reference System (EPSG:32643)
   - Bands & Dtype: 4 bands, Float32, no NaNs/Infs
   - Affine Transform: a_out = a_in / 4, e_out = e_in / 4, c and f unchanged
"""

import sys
import os
import tempfile
from pathlib import Path
import numpy as np
import rasterio

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.preprocessing import load_image_from_bytes, export_geotiff_bytes
from backend.app.services.inference import ModelRegistry


def verify_geotiff_roundtrip(input_tif_path: str = None) -> bool:
    if input_tif_path is None:
        input_tif_path = str(PROJECT_ROOT / "backend" / "sample_tiles" / "sample_real_s2.tif")

    print(f"=== Starting GeoTIFF Round-Trip Verification ===")
    print(f"Input File: {input_tif_path}")
    assert os.path.exists(input_tif_path), f"Input GeoTIFF not found: {input_tif_path}"

    # 1. Read input GeoTIFF
    with open(input_tif_path, "rb") as f:
        in_bytes = f.read()

    with rasterio.open(input_tif_path) as in_ds:
        in_crs = in_ds.crs
        in_transform = in_ds.transform
        in_bounds = in_ds.bounds
        in_res = in_ds.res
        in_count = in_ds.count
        in_width = in_ds.width
        in_height = in_ds.height
        in_dtype = in_ds.dtypes[0]

    print(f"\n[1] Input GeoTIFF Parameters:")
    print(f"    Dimensions: {in_width} x {in_height}, Bands: {in_count}, Dtype: {in_dtype}")
    print(f"    CRS: {in_crs}")
    print(f"    Resolution: {in_res[0]:.4f}m x {abs(in_res[1]):.4f}m")
    print(f"    Bounds: {in_bounds}")
    print(f"    Transform: {in_transform}")

    assert in_count == 4, f"Expected 4 bands, got {in_count}"
    assert in_crs is not None, "Input must have valid CRS"

    # 2. Ingest via preprocessing service
    img_lr, geo_meta = load_image_from_bytes(in_bytes)
    assert geo_meta["has_geo"] is True, "Failed to extract geospatial metadata"
    assert img_lr.shape == (4, in_height, in_width)

    # 3. Execute 4x Super-Resolution using Model Registry (or Bicubic baseline)
    scale_factor = 4
    registry = ModelRegistry()
    rcan_weights = str(PROJECT_ROOT / "backend" / "weights" / "rcan_best.pth")
    if os.path.exists(rcan_weights) and registry.load_model("rcan", rcan_weights):
        print("\n[2] Executing RCAN 4x Super-Resolution...")
        model = registry.get_model("rcan")
        import torch
        with torch.no_grad():
            t_in = torch.from_numpy(img_lr).unsqueeze(0).float()
            out = model(t_in)
            if isinstance(out, tuple):
                sr_img = out[0].squeeze(0).cpu().numpy()
            else:
                sr_img = out.squeeze(0).cpu().numpy()
    else:
        print("\n[2] Fallback: Executing 4x Bicubic Super-Resolution...")
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            t_in = torch.from_numpy(img_lr).unsqueeze(0).float()
            sr_img = F.interpolate(t_in, scale_factor=scale_factor, mode="bicubic", align_corners=False).squeeze(0).numpy()


    expected_w = in_width * scale_factor
    expected_h = in_height * scale_factor
    assert sr_img.shape == (4, expected_h, expected_w), f"Expected shape (4, {expected_h}, {expected_w}), got {sr_img.shape}"
    print(f"    Output array shape: {sr_img.shape}, dtype: {sr_img.dtype}")

    # 4. Export to GeoTIFF bytes
    print("\n[3] Exporting 4-Band Float32 GeoTIFF...")
    sr_geotiff_bytes = export_geotiff_bytes(sr_img, geo_metadata=geo_meta, scale_factor=scale_factor)
    assert len(sr_geotiff_bytes) > 0, "Exported GeoTIFF bytes are empty"

    # 5. Re-open with Rasterio and verify all parameters
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_file:
        tmp_file.write(sr_geotiff_bytes)
        tmp_path = tmp_file.name

    try:
        print("\n[4] Verifying Exported GeoTIFF with Rasterio:")
        with rasterio.open(tmp_path) as out_ds:
            out_crs = out_ds.crs
            out_transform = out_ds.transform
            out_bounds = out_ds.bounds
            out_res = out_ds.res
            out_count = out_ds.count
            out_width = out_ds.width
            out_height = out_ds.height
            out_data = out_ds.read()
            out_dtype = out_ds.dtypes[0]
            tags = out_ds.tags()

            print(f"    Dimensions: {out_width} x {out_height} (Expected {expected_w} x {expected_h})")
            print(f"    CRS: {out_crs} (Matches input: {str(out_crs) == str(in_crs)})")
            print(f"    Resolution: {out_res[0]:.4f}m x {abs(out_res[1]):.4f}m (Expected {in_res[0]/scale_factor:.4f}m)")
            print(f"    Bounds: {out_bounds}")
            print(f"    Transform: {out_transform}")
            print(f"    Tags: gsd='{tags.get('gsd')}', sensor='{tags.get('sensor')}'")

            # Assertions
            assert out_width == expected_w, f"Width mismatch: {out_width} != {expected_w}"
            assert out_height == expected_h, f"Height mismatch: {out_height} != {expected_h}"
            assert out_count == 4, f"Band count mismatch: {out_count} != 4"
            assert out_dtype == "float32", f"Dtype mismatch: {out_dtype} != float32"
            assert str(out_crs) == str(in_crs), f"CRS mismatch: {out_crs} != {in_crs}"

            # Resolution scaling check
            expected_res_x = in_res[0] / scale_factor
            expected_res_y = abs(in_res[1]) / scale_factor
            assert abs(out_res[0] - expected_res_x) < 1e-4, f"X resolution mismatch: {out_res[0]} != {expected_res_x}"
            assert abs(abs(out_res[1]) - expected_res_y) < 1e-4, f"Y resolution mismatch: {abs(out_res[1])} != {expected_res_y}"

            # Bounding box preservation (within sub-millimeter geographic precision)
            assert abs(out_bounds.left - in_bounds.left) < 1e-3, f"Left bound mismatch: {out_bounds.left} vs {in_bounds.left}"
            assert abs(out_bounds.top - in_bounds.top) < 1e-3, f"Top bound mismatch: {out_bounds.top} vs {in_bounds.top}"
            assert abs(out_bounds.right - in_bounds.right) < 1e-3, f"Right bound mismatch: {out_bounds.right} vs {in_bounds.right}"
            assert abs(out_bounds.bottom - in_bounds.bottom) < 1e-3, f"Bottom bound mismatch: {out_bounds.bottom} vs {in_bounds.bottom}"

            # Origin coordinate invariance in transform
            assert abs(out_transform.c - in_transform.c) < 1e-4, "Transform origin X shifted"
            assert abs(out_transform.f - in_transform.f) < 1e-4, "Transform origin Y shifted"
            assert abs(out_transform.a - in_transform.a / scale_factor) < 1e-6, "Transform pixel width not scaled"
            assert abs(out_transform.e - in_transform.e / scale_factor) < 1e-6, "Transform pixel height not scaled"

            # Radiometric sanity check
            assert not np.isnan(out_data).any(), "Exported GeoTIFF contains NaN values"
            assert not np.isinf(out_data).any(), "Exported GeoTIFF contains Inf values"
            assert (out_data >= 0.0).all(), "Negative surface reflectance values found"
            print("    Radiometric sanity: Zero NaNs, zero Infs, all reflectance >= 0.0")

        print("\n=======================================================")
        print("  ALL GEOTIFF ROUND-TRIP CHECKS PASSED SUCCESSFULLY!  ")
        print("=======================================================\n")
        return True

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    success = verify_geotiff_roundtrip()
    sys.exit(0 if success else 1)
