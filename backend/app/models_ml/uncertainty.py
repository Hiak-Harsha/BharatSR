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
    Distinguishes raw predicted uncertainty from empirically validated calibration.
    """
    if uncertainty_map.ndim == 3:
        uncertainty_map = uncertainty_map[0]

    return {
        "mean_sigma": round(float(np.mean(uncertainty_map)), 4),
        "std_sigma": round(float(np.std(uncertainty_map)), 4),
        "min_sigma": round(float(np.min(uncertainty_map)), 4),
        "max_sigma": round(float(np.max(uncertainty_map)), 4),
        "high_uncertainty_fraction": round(float(np.mean(uncertainty_map > np.percentile(uncertainty_map, 80))), 4),
        "calibration_status": "predicted_uncertainty",  # Uncalibrated until evaluate_uncertainty_calibration confirms coverage
    }


def compute_auroc_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact AUROC computation via Mann-Whitney U statistic in pure NumPy."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    pos_ranks_sum = float(np.sum(ranks[y_true == 1]))
    u_stat = pos_ranks_sum - (n_pos * (n_pos + 1)) / 2.0
    return float(np.clip(u_stat / (n_pos * n_neg), 0.0, 1.0))


def compute_auprc_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact AUPRC computation via sorted threshold integration in pure NumPy."""
    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return 0.0
    desc_order = np.argsort(-y_score)
    y_sorted = y_true[desc_order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recalls = tp / float(n_pos)
    precisions = tp / np.maximum(tp + fp, 1)
    # Average precision via step integration
    recalls = np.insert(recalls, 0, 0.0)
    precisions = np.insert(precisions, 0, precisions[0] if len(precisions) > 0 else 1.0)
    return float(np.clip(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]), 0.0, 1.0))


def evaluate_uncertainty_calibration(
    sr: np.ndarray,
    hr: np.ndarray,
    uncertainty_sigma: np.ndarray,
    high_error_quantile: float = 0.85,
) -> dict:
    """
    Rigorously evaluate uncertainty calibration against actual reconstruction error:
    - Validation NLL (Negative Log Likelihood)
    - Error vs Sigma Pearson & Spearman Rank Correlation
    - Empirical Prediction Interval Coverage (68% and 95%)
    - High-error detection AUROC and AUPRC
    - Calibration curve (quantile-binned predicted vs actual error)
    """
    from scipy.stats import spearmanr, pearsonr

    abs_diff = np.abs(sr.astype(np.float64) - hr.astype(np.float64))
    if abs_diff.ndim == 4 and uncertainty_sigma.ndim == 4 and abs_diff.shape[1] > 1 and uncertainty_sigma.shape[1] == 1:
        err = np.mean(abs_diff, axis=1, keepdims=True)
        sigma = uncertainty_sigma
    elif abs_diff.ndim == 3 and (uncertainty_sigma.ndim == 2 or (uncertainty_sigma.ndim == 3 and uncertainty_sigma.shape[0] == 1)):
        err = np.mean(abs_diff, axis=0)
        sigma = uncertainty_sigma.squeeze()
    else:
        err = abs_diff
        sigma = uncertainty_sigma

    err_flat = err.flatten()
    sigma_flat = sigma.flatten().astype(np.float64)
    assert len(err_flat) == len(sigma_flat), f"Mismatch: err {len(err_flat)} vs sigma {len(sigma_flat)}"

    # 1. Negative Log Likelihood (NLL) under Laplace heteroscedastic assumption
    sigma_safe = np.maximum(sigma_flat, 1e-6)
    nll = float(np.mean(np.log(2.0 * sigma_safe) + (err_flat / sigma_safe)))

    # 2. Correlation between predicted uncertainty and actual error
    p_corr, _ = pearsonr(sigma_flat, err_flat)
    s_corr, _ = spearmanr(sigma_flat, err_flat)

    # 3. Empirical Coverage of Prediction Intervals
    cov_1sigma = float(np.mean(err_flat <= sigma_safe))
    cov_2sigma = float(np.mean(err_flat <= (1.96 * sigma_safe)))

    # 4. High-error pixel classification AUROC & AUPRC
    high_error_thresh = float(np.percentile(err_flat, high_error_quantile * 100))
    y_true = (err_flat > high_error_thresh).astype(int)

    auroc = compute_auroc_numpy(y_true, sigma_flat)
    auprc = compute_auprc_numpy(y_true, sigma_flat)

    # 5. Calibration curve bins (10 equal-frequency quantiles)
    n_bins = 10
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(sigma_flat, quantiles)
    calib_curve = []
    for i in range(n_bins):
        mask = (sigma_flat >= bin_edges[i]) & (sigma_flat <= bin_edges[i + 1])
        if np.any(mask):
            calib_curve.append({
                "bin": i,
                "mean_predicted_sigma": round(float(np.mean(sigma_flat[mask])), 5),
                "mean_actual_error": round(float(np.mean(err_flat[mask])), 5),
                "count": int(np.sum(mask)),
            })

    # Strict scientific calibration criteria:
    # Requires positive rank correlation (> 0.25) and empirical 95% coverage between 80% and 99%
    is_calibrated = bool(s_corr > 0.25 and 0.80 <= cov_2sigma <= 0.99)

    return {
        "is_calibrated": is_calibrated,
        "calibration_status": "Physics-Calibrated Uncertainty" if is_calibrated else "Predicted Uncertainty (Uncalibrated)",
        "validation_nll": round(nll, 4),
        "mean_sigma": round(float(np.mean(sigma_flat)), 5),
        "mean_absolute_error": round(float(np.mean(err_flat)), 5),
        "pearson_correlation": round(float(p_corr), 4),
        "spearman_correlation": round(float(s_corr), 4),
        "coverage_1sigma": round(cov_1sigma, 4),
        "coverage_2sigma": round(cov_2sigma, 4),
        "coverage_68_pct": round(cov_1sigma * 100.0, 2),
        "coverage_95_pct": round(cov_2sigma * 100.0, 2),
        "high_error_threshold": round(high_error_thresh, 5),
        "auroc_high_error": round(auroc, 4),
        "auprc_high_error": round(auprc, 4),
        "high_error_auroc": round(auroc, 4),
        "high_error_auprc": round(auprc, 4),
        "calibration_curve": calib_curve,
    }

