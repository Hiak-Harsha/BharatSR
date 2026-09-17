"""
BharatSR — Uncertainty Quantification Utilities (Phase 5)
Handles variance extraction, spatial uncertainty calibration, and colormap generation.
"""

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def logvar_to_std(log_var: np.ndarray) -> np.ndarray:
    """Convert predicted log-variance to standard deviation (sigma)."""
    return np.exp(0.5 * log_var)


def generate_uncertainty_heatmap(
    uncertainty_map: np.ndarray,
    colormap: str = "magma"
) -> bytes:
    """
    Render a single-channel uncertainty map (H, W) into an RGB heatmap PNG byte string.

    Args:
        uncertainty_map: (H, W) or (1, H, W) numpy array
        colormap: Matplotlib colormap name ('magma', 'turbo', 'plasma', 'viridis')
    Returns:
        bytes of PNG image
    """
    if uncertainty_map.ndim == 3:
        uncertainty_map = uncertainty_map[0]

    # Normalize to [0, 1] using robust percentiles (2nd to 98th) to prevent outlier squashing
    p_low, p_high = np.percentile(uncertainty_map, [2, 98])
    if p_high > p_low:
        norm_map = np.clip((uncertainty_map - p_low) / (p_high - p_low), 0, 1)
    else:
        norm_map = np.zeros_like(uncertainty_map)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(norm_map)  # (H, W, 4) in [0, 1]
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def summarize_uncertainty(uncertainty_map: np.ndarray) -> dict:
    """
    Summarize spatial uncertainty distribution.
    Higher uncertainty at edges/textures indicates genuine model self-awareness.
    """
    if uncertainty_map.ndim == 3:
        uncertainty_map = uncertainty_map[0]

    return {
        "mean_sigma": float(np.mean(uncertainty_map)),
        "std_sigma": float(np.std(uncertainty_map)),
        "min_sigma": float(np.min(uncertainty_map)),
        "max_sigma": float(np.max(uncertainty_map)),
        "high_uncertainty_fraction": float(np.mean(uncertainty_map > np.percentile(uncertainty_map, 80))),
    }
