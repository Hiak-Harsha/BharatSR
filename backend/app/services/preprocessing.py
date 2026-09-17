"""
BharatSR — Preprocessing Service
Handles band loading, reflectance normalization, and tiling for inference.

CRITICAL: Physical reflectance normalization ONLY. NO ImageNet mean/std.
"""

import io
import numpy as np
from PIL import Image


def load_image_from_bytes(file_bytes: bytes, expected_bands: int = 4) -> np.ndarray:
    """
    Load an image from raw bytes (uploaded file).
    Returns (C, H, W) numpy array in reflectance scale.

    Handles:
    - Standard RGB/RGBA images (PNG, JPEG)
    - Multi-band GeoTIFF (via rasterio if available)
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

                # Product-aware reflectance normalization:
                # Sentinel-2 L2A BOA values are typically scaled by 10000 (DN 10000 = 1.0 reflectance)
                if img.max() > 10.0:
                    img = img / 10000.0
                elif img.max() > 1.5:
                    img = img / 255.0  # 8-bit imagery

                # Filter invalid / negative values, but preserve bright targets > 1.0 (clouds/snow)
                img = np.clip(img, 0.0, None)

                # Select bands (expected 4: B2, B3, B4, B8)
                if img.shape[0] > expected_bands:
                    img = img[:expected_bands]
                elif img.shape[0] < expected_bands:
                    pad = np.zeros((expected_bands - img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
                    img = np.concatenate([img, pad], axis=0)

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
    except Exception:
        pass

    # Fallback: PIL for standard image formats
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        img = np.array(pil_img, dtype=np.float32)

        if img.ndim == 2:
            # Grayscale → duplicate to expected_bands
            img = np.stack([img] * expected_bands, axis=0)
        elif img.ndim == 3:
            img = img.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)

        # Normalize
        if img.max() > 1.5:
            img = img / 255.0

        img = np.clip(img, 0, None)

        if img.shape[0] > expected_bands:
            img = img[:expected_bands]
        elif img.shape[0] < expected_bands:
            pad = np.zeros((expected_bands - img.shape[0],
                           img.shape[1], img.shape[2]), dtype=np.float32)
            img = np.concatenate([img, pad], axis=0)

        return img, None  # No geo metadata for standard images

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
    """
    if img.shape[0] >= 3:
        rgb = img[:3].transpose(1, 2, 0)  # (H, W, 3)
    else:
        rgb = np.stack([img[0]] * 3, axis=-1)

    # Clip for display only
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(rgb)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_multi_spectral_views(img: np.ndarray) -> dict:
    """
    Generate multiple spectral representations for satellite analysis:
    - RGB: True Color (Bands 4, 3, 2 -> R, G, B)
    - CIR: Color Infrared / False Color (Bands 8, 4, 3 -> NIR, R, G)
    - NDVI: Normalized Difference Vegetation Index (NIR - Red) / (NIR + Red)
    - Red, Green, Blue, NIR: Individual single-band heatmaps
    """
    import base64
    import matplotlib.pyplot as plt

    views = {}
    c, h, w = img.shape

    # 1. True Color (RGB)
    if c >= 3:
        rgb = np.clip(img[:3].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        p = Image.fromarray(rgb)
        b = io.BytesIO()
        p.save(b, format="PNG")
        views["rgb"] = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode("utf-8")

    # 2. Color Infrared (CIR) & NDVI if NIR band (band 3) exists
    if c >= 4:
        # CIR: NIR (band 3) -> R, Red (band 0) -> G, Green (band 1) -> B
        cir = np.stack([img[3], img[0], img[1]], axis=-1)
        cir = np.clip(cir * 255, 0, 255).astype(np.uint8)
        p = Image.fromarray(cir)
        b = io.BytesIO()
        p.save(b, format="PNG")
        views["cir"] = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode("utf-8")

        # NDVI: (NIR - Red) / (NIR + Red)
        nir = img[3]
        red = img[0]
        denom = nir + red + 1e-7
        ndvi = (nir - red) / denom
        # Normalize typical NDVI [-0.2, 0.85] to [0, 1]
        ndvi_norm = np.clip((ndvi + 0.2) / 1.05, 0.0, 1.0)
        cmap_ndvi = plt.get_cmap("RdYlGn")
        rgba_ndvi = (cmap_ndvi(ndvi_norm)[:, :, :3] * 255).astype(np.uint8)
        p_ndvi = Image.fromarray(rgba_ndvi)
        b_ndvi = io.BytesIO()
        p_ndvi.save(b_ndvi, format="PNG")
        views["ndvi"] = "data:image/png;base64," + base64.b64encode(b_ndvi.getvalue()).decode("utf-8")

    # 3. Individual Bands (Red, Green, Blue, NIR)
    band_names = ["red", "green", "blue", "nir"] if c >= 4 else ["band_0", "band_1", "band_2"]
    cmap_bone = plt.get_cmap("bone")
    for idx, name in enumerate(band_names[:c]):
        b_norm = np.clip(img[idx], 0.0, 1.0)
        b_colored = (cmap_bone(b_norm)[:, :, :3] * 255).astype(np.uint8)
        p = Image.fromarray(b_colored)
        b = io.BytesIO()
        p.save(b, format="PNG")
        views[name] = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode("utf-8")

    return views


def export_geotiff_bytes(img: np.ndarray, geo_metadata: dict = None, scale_factor: int = 4) -> bytes:
    """
    Export multi-band reflectance numpy array (C, H, W) to GeoTIFF bytes.
    Preserves exact float32 physical reflectance values and geospatial metadata.

    CRITICAL RULES:
    - Never fabricate coordinates or CRS (no EPSG:4326 Delhi fallbacks).
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
    # Affine(a, b, c, d, e, f): a (dx) and e (dy) are pixel resolutions; c and f are origin coordinates.
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


