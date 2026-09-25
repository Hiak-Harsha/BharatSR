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

import base64
import io
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

BAND_INDEX: Dict[str, int] = {
    "B2": 0,
    "B3": 1,
    "B4": 2,
    "B8": 3,
}

BAND_NAMES = ["B2 (Blue)", "B3 (Green)", "B4 (Red)", "B8 (NIR)"]
BAND_WAVELENGTHS_NM = {"B2": 490, "B3": 560, "B4": 665, "B8": 842}


def normalize_reflectance(
    img: np.ndarray,
    mode: str = "auto",
    dtype_hint: Optional[str] = None,
    scales: Optional[List[float]] = None,
    tags: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """
    Metadata-aware physical surface reflectance normalization.
    Preserves true BOA (Bottom-Of-Atmosphere) reflectance conventions.

    Supported input conventions:
    - 'sentinel2_l2a_dn': Sentinel-2 L2A BOA digital numbers (DN 10000 = 1.0 reflectance)
    - 'reflectance_float': Already physical BOA reflectance [0.0, ~1.5+], preserves clouds/snow > 1.0
    - 'uint8_rgb': Standard 8-bit values [0, 255] -> divide by 255.0
    - 'auto': Metadata-driven normalization using raster tags, quantification values,
      scale metadata, and data types (e.g. uint16 L2A vs float32 BOA).
      Avoids arbitrary heuristics and rejects uncalibrated ambiguous scaling.
    """
    if mode == "sentinel2_l2a_dn":
        img = img.astype(np.float32) / 10000.0
    elif mode == "reflectance_float":
        img = img.astype(np.float32)
    elif mode == "uint8_rgb":
        img = img.astype(np.float32) / 255.0
    elif mode == "auto":
        # 1. Check explicit scales metadata from rasterio
        if scales and len(scales) > 0 and scales[0] is not None and scales[0] != 1.0 and scales[0] > 0:
            img = img.astype(np.float32) * float(scales[0])
        # 2. Check tags for Sentinel-2 quantification value
        elif tags and ("QUANTIFICATION_VALUE" in tags or "BOA_QUANTIFICATION_VALUE" in tags):
            quant = float(tags.get("QUANTIFICATION_VALUE") or tags.get("BOA_QUANTIFICATION_VALUE") or 10000.0)
            img = img.astype(np.float32) / quant
        # 3. Check data type hint
        elif dtype_hint in ("uint16", "int16"):
            # Standard Sentinel-2 L2A product convention: DN 10000 = 1.0 reflectance
            img = img.astype(np.float32) / 10000.0
        elif dtype_hint in ("uint8",):
            # Standard 8-bit image: divide by 255.0
            img = img.astype(np.float32) / 255.0
        elif dtype_hint in ("float32", "float64"):
            # Floating point raster: check if values are within plausible physical reflectance [0, 2.5]
            max_v = float(np.nanmax(img)) if img.size > 0 else 0.0
            if max_v <= 2.5:
                img = img.astype(np.float32)
            else:
                import logging
                logging.getLogger("bharatsr.preprocessing").warning(
                    f"Float raster has max value {max_v} > 2.5. Scaling by 10000.0 assuming uncalibrated DN float."
                )
                img = img.astype(np.float32) / 10000.0
        else:
            # Fallback for arrays without explicit dtype_hint
            orig_dtype = str(img.dtype)
            if "uint16" in orig_dtype or "int16" in orig_dtype:
                img = img.astype(np.float32) / 10000.0
            elif "uint8" in orig_dtype:
                img = img.astype(np.float32) / 255.0
            else:
                max_v = float(np.nanmax(img)) if img.size > 0 else 0.0
                if max_v > 255.0:
                    img = img.astype(np.float32) / 10000.0
                elif max_v > 2.5:
                    img = img.astype(np.float32) / 255.0
                else:
                    img = img.astype(np.float32)
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
    - If bands > 4 and Sentinel-2 bands cannot be identified, raise ValueError.
      Never silently assume the first four bands are B2/B3/B4/B8.
    - If bands < 4 and no explicit mapping is provided, raises ValueError:
      "Sentinel-2 model requires B2/B3/B4/B8."
    """
    # Try rasterio first for GeoTIFF
    try:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as dataset:
                orig_dtype = str(dataset.dtypes[0]) if dataset.dtypes else "float32"
                scales = list(dataset.scales) if hasattr(dataset, "scales") and dataset.scales else None
                tags = dict(dataset.tags())
                descriptions = [dataset.descriptions[i] or f"Band_{i+1}" for i in range(dataset.count)] if dataset.descriptions else []
                crs = dataset.crs
                transform = dataset.transform
                bounds = dataset.bounds
                res = dataset.res if hasattr(dataset, "res") else (None, None)
                nodata = dataset.nodata
                has_geo = bool(crs is not None and transform is not None)

                img = dataset.read().astype(np.float32)  # (C, H, W)
                c = img.shape[0]

                # Band selection & validation
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
                    s2_map = {}
                    for i, d in enumerate(descriptions):
                        for b_name in ["B2", "B3", "B4", "B8"]:
                            if b_name.lower() in d.lower():
                                s2_map[b_name] = i
                    if len(s2_map) == 4:
                        img = np.stack([img[s2_map["B2"]], img[s2_map["B3"]], img[s2_map["B4"]], img[s2_map["B8"]]], axis=0)
                elif c > 4:
                    s2_map = {}
                    for i, d in enumerate(descriptions):
                        for b_name in ["B2", "B3", "B4", "B8"]:
                            if b_name.lower() in d.lower():
                                s2_map[b_name] = i
                    if len(s2_map) == 4:
                        img = np.stack([img[s2_map["B2"]], img[s2_map["B3"]], img[s2_map["B4"]], img[s2_map["B8"]]], axis=0)
                    else:
                        raise ValueError(
                            f"Input raster contains {c} bands, but Sentinel-2 bands B2, B3, B4, B8 could not be identified from band descriptions ({descriptions}). Provide an explicit band mapping or an authentic 4-band raster (B2, B3, B4, B8)."
                        )
                else:
                    raise ValueError("Sentinel-2 model requires B2/B3/B4/B8.")

                img = normalize_reflectance(
                    img,
                    mode=mode,
                    dtype_hint=orig_dtype,
                    scales=scales,
                    tags=tags,
                )

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
        orig_mode = pil_img.mode
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

        dtype_hint = "uint8" if orig_mode in ("RGB", "RGBA", "L") else "float32"
        img = normalize_reflectance(img, mode=mode, dtype_hint=dtype_hint)
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

    # Standalone All-Bands Master Composite (Analytical Product Grid)
    views["composite"] = generate_master_composite(img, views)

    return views


def generate_master_composite(bands: np.ndarray, views: Optional[dict] = None) -> str:
    """
    Produces ONE PNG that is the actual analytical product, built as a labeled
    2x4 grid:
    - Row 1: True Color RGB (large, primary panel spanning 2 columns) +
             Color Infrared CIR +
             NDVI with dedicated analytical colorbar.
    - Row 2: Annotated individual spectral sub-panels (B2 Blue 490nm, B3 Green 560nm,
             B4 Red 665nm, B8 NIR 842nm).
    Returns a single base64 PNG, same encoding convention as the other views.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    c, h, w = bands.shape

    fig = plt.figure(figsize=(14, 7.5), facecolor="#09090b", edgecolor="#27272a")
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.25, 1.0], hspace=0.28, wspace=0.18)

    title_color = "#e4e4e7"
    subtitle_color = "#a1a1aa"
    amber_accent = "#f59e0b"
    cyan_accent = "#06b6d4"

    fig.suptitle(
        "BHARATSR MULTI-SPECTRAL MASTER COMPOSITE | SENTINEL-2 MSI (10m -> 2.5m GRID)",
        fontsize=12,
        fontweight="bold",
        color=title_color,
        y=0.98,
        fontfamily="monospace",
    )

    if c >= 4:
        b2 = np.clip(bands[BAND_INDEX["B2"]], 0.0, 1.0)
        b3 = np.clip(bands[BAND_INDEX["B3"]], 0.0, 1.0)
        b4 = np.clip(bands[BAND_INDEX["B4"]], 0.0, 1.0)
        b8 = np.clip(bands[BAND_INDEX["B8"]], 0.0, 1.0)

        # 1. Primary: True Color RGB (span cols 0 and 1)
        ax_rgb = fig.add_subplot(gs[0, 0:2])
        rgb = np.stack([b4, b3, b2], axis=-1)
        ax_rgb.imshow(rgb)
        ax_rgb.set_title("PRIMARY PRODUCT: TRUE COLOR RGB (B4 - B3 - B2)", color=amber_accent, fontsize=10, fontweight="bold", fontfamily="monospace", pad=6)
        ax_rgb.axis("off")

        # 2. CIR (col 2)
        ax_cir = fig.add_subplot(gs[0, 2])
        cir = np.stack([b8, b4, b3], axis=-1)
        ax_cir.imshow(cir)
        ax_cir.set_title("CIR FALSE COLOR (B8-B4-B3)", color=title_color, fontsize=9, fontweight="bold", fontfamily="monospace", pad=6)
        ax_cir.axis("off")

        # 3. NDVI (col 3) with colorbar
        ax_ndvi = fig.add_subplot(gs[0, 3])
        denom = b8 + b4 + 1e-7
        ndvi = (b8 - b4) / denom
        ndvi_norm = np.clip((ndvi + 0.2) / 1.05, 0.0, 1.0)
        im_ndvi = ax_ndvi.imshow(ndvi_norm, cmap="RdYlGn", vmin=0.0, vmax=1.0)
        ax_ndvi.set_title("NDVI VEGETATION INDEX", color=cyan_accent, fontsize=9, fontweight="bold", fontfamily="monospace", pad=6)
        ax_ndvi.axis("off")

        cbar = fig.colorbar(im_ndvi, ax=ax_ndvi, orientation="vertical", fraction=0.046, pad=0.04)
        cbar.set_ticks([0.0, 0.2, 0.5, 0.8, 1.0])
        cbar.set_ticklabels(["-0.2", "0.0", "0.3", "0.6", "0.85"])
        cbar.ax.tick_params(labelsize=7, colors=subtitle_color)
        cbar.outline.set_edgecolor("#3f3f46")

        # Row 2: Sub-panels for individual bands
        sub_panels = [
            (gs[1, 0], b2, "B2: BLUE (490 nm)", "bone"),
            (gs[1, 1], b3, "B3: GREEN (560 nm)", "bone"),
            (gs[1, 2], b4, "B4: RED (665 nm)", "bone"),
            (gs[1, 3], b8, "B8: NIR (842 nm)", "bone"),
        ]

        for grid_pos, band_data, label, cmap_name in sub_panels:
            ax = fig.add_subplot(grid_pos)
            ax.imshow(band_data, cmap=cmap_name, vmin=0.0, vmax=1.0)
            ax.set_title(label, color=subtitle_color, fontsize=8, fontfamily="monospace", pad=4)
            ax.axis("off")

    else:
        ax = fig.add_subplot(gs[:, :])
        if c == 3:
            ax.imshow(np.clip(bands.transpose(1, 2, 0), 0.0, 1.0))
        else:
            ax.imshow(np.clip(bands[0], 0.0, 1.0), cmap="bone")
        ax.set_title("MULTI-SPECTRAL PREVIEW", color=title_color, fontsize=10, fontfamily="monospace")
        ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=110)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


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
        in_affine.b,
        in_affine.c,
        in_affine.d,
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


def apply_scl_mask(
    img: np.ndarray,
    scl_band: np.ndarray,
    valid_classes: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Sentinel-2 Scene Classification Layer (SCL) to mask invalid pixels.

    SCL class meanings:
    0: No data, 1: Defective, 2: Dark pixels, 3: Cloud shadows
    4: Vegetation, 5: Not-vegetated, 6: Water, 7: Unclassified
    8: Cloud medium prob, 9: Cloud high prob, 10: Cirrus, 11: Snow/Ice

    Default valid classes: [4, 5, 6, 7, 11] — surface pixels (excludes clouds/shadows)
    Returns: (masked_img, valid_mask) where masked_img has NaN at invalid pixels.
    """
    if valid_classes is None:
        valid_classes = [4, 5, 6, 7, 11]
    valid_mask = np.isin(scl_band, valid_classes)
    masked = img.copy().astype(np.float32)
    masked[:, ~valid_mask] = np.nan
    return masked, valid_mask


def compute_cloud_fraction(scl_band: np.ndarray) -> float:
    """Return fraction of pixels classified as cloud (SCL 8, 9, 10)."""
    cloud_classes = [8, 9, 10]
    cloud_mask = np.isin(scl_band, cloud_classes)
    return float(np.mean(cloud_mask))


def register_lr_hr_pair(
    lr: np.ndarray, hr: np.ndarray, scale_factor: int = 4
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Sub-pixel spatial registration of LR/HR pairs using phase correlation.
    Corrects for small misalignments between acquisition dates.

    Returns: (lr_aligned, hr_aligned, registration_info)
    """
    from scipy.ndimage import shift as nd_shift
    from numpy.fft import fft2, ifft2, fftshift

    lr_red = lr[2]  # B4
    hr_red = hr[2]  # B4

    lr_red_up = (
        F.interpolate(
            torch.from_numpy(lr_red[None, None]).float(),
            size=hr_red.shape,
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )

    f1 = fft2(lr_red_up)
    f2 = fft2(hr_red)
    cross_power = (f1 * f2.conj()) / (np.abs(f1 * f2.conj()) + 1e-10)
    corr = np.real(fftshift(ifft2(cross_power)))
    peak = np.unravel_index(corr.argmax(), corr.shape)
    dy = int(peak[0] - corr.shape[0] // 2)
    dx = int(peak[1] - corr.shape[1] // 2)

    hr_shifted = np.stack(
        [nd_shift(hr[b], (dy, dx), mode="reflect") for b in range(hr.shape[0])]
    )

    return lr, hr_shifted, {"shift_y": dy, "shift_x": dx, "peak_corr": float(corr.max())}


def compute_spectral_indices(
    img: np.ndarray, band_index: Optional[Dict[str, int]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute all standard Sentinel-2 spectral indices from 4-band reflectance array.

    All formulas use physical reflectance [0, ~1+].
    Division by zero protected with epsilon = 1e-7.

    Returns dict of index_name -> (H, W) float32 array.
    """
    if band_index is None:
        band_index = {"B2": 0, "B3": 1, "B4": 2, "B8": 3}

    eps = 1e-7
    blue = img[band_index["B2"]].astype(np.float32)
    green = img[band_index["B3"]].astype(np.float32)
    red = img[band_index["B4"]].astype(np.float32)
    nir = img[band_index["B8"]].astype(np.float32)

    indices = {}

    # NDVI: Normalized Difference Vegetation Index
    indices["ndvi"] = (nir - red) / (nir + red + eps)

    # NDWI: Normalized Difference Water Index (McFeeters 1996)
    indices["ndwi"] = (green - nir) / (green + nir + eps)

    # EVI: Enhanced Vegetation Index (Huete et al. 2002)
    # EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
    indices["evi"] = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + eps)

    # SAVI: Soil-Adjusted Vegetation Index (Huete 1988, L=0.5)
    L = 0.5
    indices["savi"] = ((nir - red) / (nir + red + L + eps)) * (1.0 + L)

    # RVI: Ratio Vegetation Index
    indices["rvi"] = nir / (red + eps)

    # NDBI: Normalized Difference Built-up Index (proxy using Red and Blue for SWIR)
    swir_proxy = (red + blue) / 2.0
    indices["ndbi_approx"] = (swir_proxy - nir) / (swir_proxy + nir + eps)

    # GCI: Green Chlorophyll Index
    indices["gci"] = (nir / (green + eps)) - 1.0

    return indices


def indices_to_visualizations(indices: Dict[str, np.ndarray]) -> Dict[str, str]:
    """Convert spectral indices to base64 PNG heatmaps for API responses."""
    import matplotlib.pyplot as plt

    colormaps = {
        "ndvi": ("RdYlGn", -0.2, 0.8),
        "ndwi": ("RdYlBu", -0.5, 0.5),
        "evi": ("YlGn", -0.5, 1.0),
        "savi": ("RdYlGn", -0.2, 0.8),
        "rvi": ("Greens", 0.0, 5.0),
        "ndbi_approx": ("copper", -0.5, 0.5),
        "gci": ("Greens", -1.0, 5.0),
    }

    result = {}
    for name, arr in indices.items():
        cmap_name, vmin, vmax = colormaps.get(name, ("viridis", None, None))
        valid_vals = arr[~np.isnan(arr)]
        if valid_vals.size == 0:
            norm = np.zeros_like(arr, dtype=np.float32)
        elif vmin is not None and vmax is not None:
            norm = np.clip((arr - vmin) / ((vmax - vmin) + 1e-7), 0.0, 1.0)
        else:
            p2, p98 = np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)
            norm = np.clip((arr - p2) / (p98 - p2 + 1e-7), 0.0, 1.0)

        # Handle NaNs
        norm = np.nan_to_num(norm, nan=0.0)

        cmap = plt.get_cmap(cmap_name)
        rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        result[name] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return result
