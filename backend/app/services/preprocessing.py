"""
BharatSR — Preprocessing Service
Handles band loading, reflectance normalization, and tiling for inference.

CRITICAL: Physical reflectance normalization ONLY. NO ImageNet mean/std.
Canonical Sentinel-2 4-Band Input Order:
- B2 (Blue, 490nm)  -> Index 0
- B3 (Green, 560nm) -> Index 1
- B4 (Red, 665nm)   -> Index 2
- B8 (NIR, 842nm)   -> Index 3
"""

import io
from typing import Dict, Optional, Tuple
import numpy as np
from PIL import Image

BAND_INDEX: Dict[str, int] = {
    "B2": 0,
    "B3": 1,
    "B4": 2,
    "B8": 3,
}

BAND_NAMES = ["B2 (Blue)", "B3 (Green)", "B4 (Red)", "B8 (NIR)"]
BAND_WAVELENGTHS_NM = {"B2": 490, "B3": 560, "B4": 665, "B8": 842}


def normalize_reflectance(img: np.ndarray, mode: str = "auto") -> np.ndarray:
    """
    Product-aware physical surface reflectance normalization.
    Explicit input modes:
    - 'sentinel2_l2a_dn': Sentinel-2 L2A BOA digital numbers (DN 10000 = 1.0 reflectance)
    - 'reflectance_float': Already physical BOA reflectance [0.0, ~1+], preserves clouds/snow > 1.0
    - 'uint8_rgb': Standard 8-bit values [0, 255] -> divide by 255.0
    - 'auto': Inspects data range and dtype to choose appropriate scaling
    """
    img = img.astype(np.float32)
    if mode == "sentinel2_l2a_dn":
        img = img / 10000.0
    elif mode == "reflectance_float":
        pass  # Keep as-is
    elif mode == "uint8_rgb":
        img = img / 255.0
    elif mode == "auto":
        max_val = float(np.nanmax(img)) if img.size > 0 else 0.0
        if max_val > 10.0:
            # Scaled DN (e.g. S2 L2A 0-10000+)
            img = img / 10000.0
        elif max_val > 1.5:
            # 8-bit range (0-255)
            img = img / 255.0
        # Otherwise already float reflectance [0, ~1.5]
    else:
        raise ValueError(
            f"Unknown reflectance normalization mode: '{mode}'. "
            f"Expected 'sentinel2_l2a_dn', 'reflectance_float', 'uint8_rgb', or 'auto'."
        )

    # Filter invalid negative values, but preserve bright targets > 1.0 (clouds, snow)
    img = np.clip(img, 0.0, None)
    return img


def load_image_from_bytes(
    file_bytes: bytes,
    expected_bands: int = 4,
    mode: str = "auto",
    band_mapping: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, Optional[dict]]:
    """
    Load an image from raw bytes (uploaded file).
    Returns (C, H, W) numpy array in reflectance scale and geo_meta dict.

    Handles:
    - Multi-band GeoTIFF (via rasterio)
    - Standard image formats (PNG, JPEG)

    CRITICAL RULES:
    - Sentinel-2 model requires B2, B3, B4, B8.
    - Never pad arbitrary RGB images with zero channels.
    - If bands < 4 and no explicit mapping is provided, raises ValueError:
      "Sentinel-2 model requires B2/B3/B4/B8."
    """
    # Try rasterio first for GeoTIFF
    try:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as dataset:
                img = dataset.read().astype(np.float32)  # (C, H, W)
                crs = dataset.crs
                transform = dataset.transform
                bounds = dataset.bounds
                res = dataset.res if hasattr(dataset, "res") else (None, None)
                nodata = dataset.nodata
                descriptions = [dataset.descriptions[i] or f"Band_{i+1}" for i in range(dataset.count)] if dataset.descriptions else []
                tags = dict(dataset.tags())
                has_geo = bool(crs is not None and transform is not None)

                # Band selection & validation
                c = img.shape[0]
                if band_mapping is not None:
                    selected = []
                    for b_key in ["B2", "B3", "B4", "B8"]:
                        if b_key not in band_mapping:
                            raise ValueError("Sentinel-2 model requires B2/B3/B4/B8.")
                        src_idx = band_mapping[b_key]
                        if src_idx >= c:
                            raise ValueError(f"Band index {src_idx} out of range for {c}-band image.")
                        selected.append(img[src_idx])
                    img = np.stack(selected, axis=0)
                elif c == 4:
                    # Exactly 4 bands: standard Sentinel-2 subset [B2, B3, B4, B8]
                    pass
                elif c > 4:
                    s2_map = {}
                    for i, d in enumerate(descriptions):
                        for b_name in ["B2", "B3", "B4", "B8"]:
                            if b_name.lower() in d.lower():
                                s2_map[b_name] = i
                    if len(s2_map) == 4:
                        img = np.stack([img[s2_map["B2"]], img[s2_map["B3"]], img[s2_map["B4"]], img[s2_map["B8"]]], axis=0)
                    else:
                        img = img[:expected_bands]
                else:
                    # Fewer than 4 bands and no mapping -> REJECT (do not pad with zeros!)
                    raise ValueError("Sentinel-2 model requires B2/B3/B4/B8.")

                img = normalize_reflectance(img, mode=mode)

                geo_meta = {
                    "has_geo": has_geo,
                    "crs": str(crs) if crs else None,
                    "transform": list(transform) if transform else None,
                    "width": dataset.width,
                    "height": dataset.height,
                    "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top] if bounds else None,
                    "pixel_size": [float(res[0]), float(res[1])] if res[0] is not None else None,
                    "nodata": float(nodata) if nodata is not None else None,
                    "descriptions": descriptions,
                    "tags": tags,
                    "message": "Geospatially referenced" if has_geo else "No geospatial reference available",
                }
                return img, geo_meta
    except ValueError:
        raise
    except Exception:
        pass

    # Fallback: Standard image formats (PNG, JPEG)
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        img = np.array(pil_img, dtype=np.float32)

        if img.ndim == 3:
            img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        elif img.ndim == 2:
            img = img[np.newaxis, ...]  # (1, H, W)

        c = img.shape[0]
        if band_mapping is not None:
            selected = []
            for b_key in ["B2", "B3", "B4", "B8"]:
                if b_key not in band_mapping:
                    raise ValueError("Sentinel-2 model requires B2/B3/B4/B8.")
                src_idx = band_mapping[b_key]
                if src_idx >= c:
                    raise ValueError(f"Band index {src_idx} out of range for {c}-band image.")
                selected.append(img[src_idx])
            img = np.stack(selected, axis=0)
        elif c == 4:
            pass
        else:
            # Reject arbitrary RGB / Grayscale without explicit mapping: DO NOT ZERO-PAD!
            raise ValueError("Sentinel-2 model requires B2/B3/B4/B8.")

        img = normalize_reflectance(img, mode=mode)
        return img, None

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not load image: {e}")


def load_sample_tile(sample_path: str) -> tuple:
    """Load a pre-saved sample tile (.npz)."""
    data = np.load(sample_path)
    lr = data["lr"].astype(np.float32)
    hr = data.get("hr", None)
    if hr is not None:
        hr = hr.astype(np.float32)
    return lr, hr


def prepare_for_inference(img: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    """
    Prepare an image for model inference.
    Ensures correct shape and dtype.

    Args:
        img: (C, H, W) numpy array in reflectance scale
    Returns:
        (C, H, W) float32 numpy array ready for model
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    return img


def numpy_to_png_bytes(img: np.ndarray) -> bytes:
    """
    Convert (C, H, W) reflectance array to PNG bytes for API response.
    Clips to [0, 1] for display ONLY — the data itself is not modified.
    For 4-band Sentinel-2 [B2, B3, B4, B8]: True Color is (B4, B3, B2).
    """
    c = img.shape[0]
    if c >= 4:
        rgb = np.stack([
            img[BAND_INDEX["B4"]],
            img[BAND_INDEX["B3"]],
            img[BAND_INDEX["B2"]],
        ], axis=-1)
    elif c == 3:
        rgb = img.transpose(1, 2, 0)
    else:
        rgb = np.stack([img[0]] * 3, axis=-1)

    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(rgb)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_multi_spectral_views(img: np.ndarray) -> dict:
    """
    Generate multiple spectral representations for satellite analysis:
    - RGB: True Color (B4 Red, B3 Green, B2 Blue)
    - CIR: Color Infrared / False Color (B8 NIR, B4 Red, B3 Green)
    - NDVI: Normalized Difference Vegetation Index (B8 - B4) / (B8 + B4 + 1e-7)
    - B2, B3, B4, B8: Individual single-band heatmaps
    """
    import base64
    import matplotlib.pyplot as plt

    views = {}
    c, h, w = img.shape

    # 1. True Color (RGB: R=B4, G=B3, B=B2)
    if c >= 4:
        rgb = np.stack([
            img[BAND_INDEX["B4"]],
            img[BAND_INDEX["B3"]],
            img[BAND_INDEX["B2"]],
        ], axis=-1)
    elif c == 3:
        rgb = img.transpose(1, 2, 0)
    else:
        rgb = np.stack([img[0]] * 3, axis=-1)

    rgb_disp = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    p = Image.fromarray(rgb_disp)
    b = io.BytesIO()
    p.save(b, format="PNG")
    views["rgb"] = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode("utf-8")

    # 2. Color Infrared (CIR: R=B8, G=B4, B=B3) & NDVI if at least 4 bands
    if c >= 4:
        cir = np.stack([
            img[BAND_INDEX["B8"]],
            img[BAND_INDEX["B4"]],
            img[BAND_INDEX["B3"]],
        ], axis=-1)
        cir_disp = np.clip(cir * 255, 0, 255).astype(np.uint8)
        p_cir = Image.fromarray(cir_disp)
        b_cir = io.BytesIO()
        p_cir.save(b_cir, format="PNG")
        views["cir"] = "data:image/png;base64," + base64.b64encode(b_cir.getvalue()).decode("utf-8")

        # NDVI: (B8 - B4) / (B8 + B4 + 1e-7)
        nir = img[BAND_INDEX["B8"]]
        red = img[BAND_INDEX["B4"]]
        denom = nir + red + 1e-7
        ndvi = (nir - red) / denom
        ndvi_norm = np.clip((ndvi + 0.2) / 1.05, 0.0, 1.0)
        cmap_ndvi = plt.get_cmap("RdYlGn")
        rgba_ndvi = (cmap_ndvi(ndvi_norm)[:, :, :3] * 255).astype(np.uint8)
        p_ndvi = Image.fromarray(rgba_ndvi)
        b_ndvi = io.BytesIO()
        p_ndvi.save(b_ndvi, format="PNG")
        views["ndvi"] = "data:image/png;base64," + base64.b64encode(b_ndvi.getvalue()).decode("utf-8")

    # 3. Individual Bands (B2 Blue, B3 Green, B4 Red, B8 NIR)
    band_keys = ["B2", "B3", "B4", "B8"] if c >= 4 else [f"band_{i}" for i in range(c)]
    cmap_bone = plt.get_cmap("bone")
    for idx, key in enumerate(band_keys[:c]):
        b_norm = np.clip(img[idx], 0.0, 1.0)
        b_colored = (cmap_bone(b_norm)[:, :, :3] * 255).astype(np.uint8)
        p_b = Image.fromarray(b_colored)
        buf_b = io.BytesIO()
        p_b.save(buf_b, format="PNG")
        name = key.lower() if key in ["B2", "B3", "B4", "B8"] else f"band_{idx}"
        views[name] = "data:image/png;base64," + base64.b64encode(buf_b.getvalue()).decode("utf-8")

    # Map standard named aliases for frontend tabs:
    if c >= 4:
        views["blue"] = views.get("b2", views.get("band_0"))
        views["green"] = views.get("b3", views.get("band_1"))
        views["red"] = views.get("b4", views.get("band_2"))
        views["nir"] = views.get("b8", views.get("band_3"))

    return views


def export_geotiff_bytes(img: np.ndarray, geo_metadata: dict = None, scale_factor: int = 4) -> bytes:
    """
    Export multi-band reflectance numpy array (C, H, W) to GeoTIFF bytes.
    Preserves exact float32 physical reflectance values and geospatial metadata.

    CRITICAL RULES:
    - Never fabricate coordinates or CRS.
    - If no geospatial metadata exists, raises ValueError stating 'No geospatial reference available'.
    - For 4x SR: output pixel size = input pixel size / scale_factor; affine transform is scaled accordingly.
    """
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine

    if not geo_metadata or not geo_metadata.get("has_geo", False):
        raise ValueError("No geospatial reference available. Authoritative GeoTIFF export requires georeferenced raster metadata.")

    crs_str = geo_metadata.get("crs")
    if not crs_str or crs_str == "None":
        raise ValueError("No geospatial reference available (missing CRS).")

    raw_transform = geo_metadata.get("transform")
    if not raw_transform or len(raw_transform) < 6:
        raise ValueError("No geospatial reference available (missing affine transform).")

    # Recalculate affine transform for SR:
    # Input pixel size = p -> Output pixel size = p / scale_factor
    in_affine = Affine(*raw_transform[:6])
    out_affine = Affine(
        in_affine.a / scale_factor,
        in_affine.b / scale_factor,
        in_affine.c,
        in_affine.d / scale_factor,
        in_affine.e / scale_factor,
        in_affine.f,
    )

    c, h, w = img.shape
    nodata_val = geo_metadata.get("nodata", None)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=h,
            width=w,
            count=c,
            dtype=np.float32,
            crs=crs_str,
            transform=out_affine,
            nodata=nodata_val,
        ) as dataset:
            for b in range(c):
                dataset.write(img[b].astype(np.float32), b + 1)

            band_names = ["B2 - Blue (490nm)", "B3 - Green (560nm)", "B4 - Red (665nm)", "B8 - NIR (842nm)"]
            for b in range(min(c, len(band_names))):
                dataset.set_band_description(b + 1, band_names[b])

            dataset.update_tags(
                sensor="Sentinel-2 MSI",
                processing=f"BharatSR {scale_factor}x Super-Resolution",
                gsd=f"{abs(out_affine.a):.2f}m-equivalent output grid",
                spectral_bands="B2, B3, B4, B8 (Selected 4-band subset of Sentinel-2)",
            )
        return bytes(memfile.getbuffer())
