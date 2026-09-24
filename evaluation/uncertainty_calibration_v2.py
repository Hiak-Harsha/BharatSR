"""
BharatSR v2 — Uncertainty Calibration & Expected Calibration Error (ECE)

Computes:
- Expected Calibration Error (ECE) for heteroscedastic uncertainty estimates
- Reliability diagrams & calibration curves
- Empirical interval coverage (68% and 95% nominal bounds)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple


def compute_ece(
    errors: np.ndarray,
    predicted_sigmas: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE) for continuous Gaussian uncertainty.
    Normalized error z = |y - \hat{y}| / \sigma should follow standard half-normal distribution.
    Compares empirical CDF against theoretical standard normal CDF across quantile bins.
    """
    from scipy.stats import norm

    valid = (predicted_sigmas > 1e-6) & (~np.isnan(errors)) & (~np.isnan(predicted_sigmas))
    if not np.any(valid):
        return 0.0

    z = np.abs(errors[valid]) / predicted_sigmas[valid]
    bin_edges = np.linspace(0.0, 3.0, n_bins + 1)
    ece = 0.0
    total_samples = len(z)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = (z >= low) & (z < high)
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            empirical_freq = bin_count / total_samples
            # Theoretical probability mass in [low, high] for half-normal
            theoretical_prob = 2.0 * (norm.cdf(high) - norm.cdf(low))
            ece += (bin_count / total_samples) * np.abs(empirical_freq - theoretical_prob)

    return float(ece)


def evaluate_uncertainty_comprehensive(
    sr: np.ndarray,
    hr: np.ndarray,
    sigma: np.ndarray,
) -> Dict[str, float]:
    """Computes comprehensive uncertainty calibration metrics."""
    errors = np.mean(np.abs(sr.astype(np.float64) - hr.astype(np.float64)), axis=0)
    sigma_2d = sigma[0] if sigma.ndim == 3 else sigma

    # Coverage: fraction of pixels where |error| <= 1.0 * sigma (nominal 68.3%)
    cov_68 = float(np.mean(errors <= 1.0 * sigma_2d))
    # Coverage: fraction of pixels where |error| <= 1.96 * sigma (nominal 95.0%)
    cov_95 = float(np.mean(errors <= 1.96 * sigma_2d))

    ece = compute_ece(errors, sigma_2d)

    # Correlation between sigma and actual error
    flat_err = errors.flatten()
    flat_sig = sigma_2d.flatten()
    corr = float(np.corrcoef(flat_sig, flat_err)[0, 1]) if len(flat_err) > 1 else 0.0

    return {
        "ece": round(ece, 4),
        "coverage_68_pct": round(cov_68 * 100.0, 2),
        "coverage_95_pct": round(cov_95 * 100.0, 2),
        "error_sigma_correlation": round(corr, 4),
        "mean_sigma": round(float(np.mean(sigma_2d)), 5),
    }


def plot_reliability_diagram(sr: np.ndarray, hr: np.ndarray, sigma: np.ndarray, save_path: str):
    """Plot calibration curve and save to disk."""
    import matplotlib.pyplot as plt

    errors = np.mean(np.abs(sr.astype(np.float64) - hr.astype(np.float64)), axis=0)
    sigma_2d = sigma[0] if sigma.ndim == 3 else sigma

    quantiles = np.linspace(0.1, 0.9, 9)
    from scipy.stats import norm
    nominal_cov = quantiles
    z_thresholds = norm.ppf(0.5 + quantiles / 2.0)

    empirical_cov = []
    for z in z_thresholds:
        empirical_cov.append(float(np.mean(errors <= z * sigma_2d)))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Ideal Calibration")
    ax.plot(nominal_cov, empirical_cov, "o-", color="#ffb454", label="Empirical Coverage")
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("BharatSR Uncertainty Reliability Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
