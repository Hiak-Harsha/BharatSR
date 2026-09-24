"""
BharatSR — Loss Functions and Metrics (Phase 2+)

Loss functions:
- L1 reconstruction loss (Phase 2 baseline)
- SAM spectral loss (Phase 5)
- Canonical area-averaging downsample consistency loss (Phase 5)
- Heteroscedastic Gaussian NLL loss for uncertainty quantification (Phase 5)

Metrics:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- SAM (Spectral Angle Mapper) — measures spectral fidelity
- Downsample consistency error (MAE)
- Spectral MAE
- Edge-based diagnostics: false_edge_rate, missing_edge_rate, high_frequency_excess_rate

CRITICAL: All metrics operate on physical reflectance values [0, ~1+].
No ImageNet normalization anywhere.
"""

from typing import Optional, Tuple, Dict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================
# CANONICAL DEGRADATION OPERATOR
# ========================

def degrade_canonical_4x(x, scale_factor: int = 4):
    """
    Canonical area-averaging degradation assumption D(x):
    Exact 4x4 area average aligned with the LR grid.
    Described strictly as the 'canonical area-averaging degradation assumption'
    (not 'exact optical physics', which requires sensor PSF/MTF modeling).

    Used identically for:
    - Training consistency loss
    - Validation consistency evaluation
    - Scientific benchmarking
    - Automated tests

    Supports:
    - torch.Tensor of shape (B, C, H, W) or (C, H, W)
    - numpy.ndarray of shape (C, H, W)
    """
    if isinstance(x, torch.Tensor):
        is_3d = (x.ndim == 3)
        if is_3d:
            x = x.unsqueeze(0)
        # 4x4 area average box filter
        degraded = F.avg_pool2d(x, kernel_size=scale_factor, stride=scale_factor)
        return degraded.squeeze(0) if is_3d else degraded
    elif isinstance(x, np.ndarray):
        c, h, w = x.shape
        h_out, w_out = h // scale_factor, w // scale_factor
        trimmed = x[:, :h_out * scale_factor, :w_out * scale_factor]
        degraded = trimmed.reshape(c, h_out, scale_factor, w_out, scale_factor).mean(axis=(2, 4))
        return degraded.astype(np.float32)
    else:
        raise TypeError(f"Unsupported type for degradation operator: {type(x)}")


# ========================
# LOSS FUNCTIONS
# ========================

class ReconstructionLoss(nn.Module):
    """L1 reconstruction loss on physical reflectance values."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return self.l1(sr, hr)


class SpectralConsistencyLoss(nn.Module):
    """
    Downsample Consistency Loss (L_DC):
    L_DC = MAE(D(SR), LR)
    Degrades SR output using the canonical 4x4 area average operator and compares against LR.
    Does NOT use bilinear interpolation.
    """

    def __init__(self, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, sr: torch.Tensor, lr_original: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sr: (B, C, H_sr, W_sr) super-resolved output
            lr_original: (B, C, H_lr, W_lr) original LR sensor input
        """
        sr_down = degrade_canonical_4x(sr, scale_factor=self.scale_factor)
        # Crop or match if dimensions differ slightly due to rounding
        if sr_down.shape[2:] != lr_original.shape[2:]:
            min_h = min(sr_down.shape[2], lr_original.shape[2])
            min_w = min(sr_down.shape[3], lr_original.shape[3])
            sr_down = sr_down[:, :, :min_h, :min_w]
            lr_original = lr_original[:, :, :min_h, :min_w]
        return F.l1_loss(sr_down, lr_original)


class SAMLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) Loss:
    L_SAM = mean(arccos(cosine_similarity(SR_pixel, HR_pixel)))

    Numerically stabilized:
    - Avoids NaN gradients near cos = +/- 1 via margin clamping
    - Handles near-zero spectral vectors by masking uninformative low-reflectance pixels
    """

    def __init__(self, eps: float = 1e-7, clamp_margin: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.clamp_margin = clamp_margin

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # sr, hr: (B, C, H, W)
        dot = torch.sum(sr * hr, dim=1)  # (B, H, W)
        norm_sr = torch.norm(sr, p=2, dim=1)  # (B, H, W)
        norm_hr = torch.norm(hr, p=2, dim=1)  # (B, H, W)

        denom = norm_sr * norm_hr
        valid = (norm_sr > self.eps) & (norm_hr > self.eps)

        cos_sim = torch.ones_like(dot)
        cos_sim[valid] = dot[valid] / (denom[valid] + self.eps)

        # Gradient-safe arccos clamping
        cos_clamped = torch.clamp(cos_sim, -1.0 + self.clamp_margin, 1.0 - self.clamp_margin)
        sam_map = torch.acos(cos_clamped)

        return sam_map.mean()


class UncertaintyLoss(nn.Module):
    """
    Heteroscedastic negative log-likelihood (NLL) loss for uncertainty quantification.
    Statistically consistent Gaussian formulation:
    s = log(sigma^2)
    L_unc = 0.5 * exp(-s) * (HR - SR)^2 + 0.5 * s
    """

    def __init__(self, mode: str = "gaussian"):
        super().__init__()
        self.mode = mode

    def forward(self, sr: torch.Tensor, log_var: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Clamp log_var to prevent numerical explosion or collapse
        log_var = torch.clamp(log_var, -6.0, 6.0)
        precision = torch.exp(-log_var)

        if self.mode == "gaussian":
            # Statistically consistent Gaussian heteroscedastic loss:
            # 0.5 * exp(-s) * (HR - SR)^2 + 0.5 * s
            diff = ((hr - sr) ** 2).mean(dim=1, keepdim=True)
            loss = 0.5 * precision * diff + 0.5 * log_var
        else:
            # Laplace formulation: 0.5 * exp(-s) * |HR - SR| + 0.5 * s
            diff = torch.abs(hr - sr).mean(dim=1, keepdim=True)
            loss = 0.5 * precision * diff + 0.5 * log_var

        return loss.mean()


class LPIPSSpectralLoss(nn.Module):
    """
    Adapted LPIPS (Learned Perceptual Image Patch Similarity) for 4-band satellite imagery.

    Uses VGG16 feature space on the 3-band RGB subset (B4, B3, B2) for perceptual quality.
    The NIR band (B8) is evaluated separately via L1 loss (no perceptual VGG equivalent).

    NOTE: Falls back gracefully to gradient similarity if torchvision is unavailable.
    """

    def __init__(self, use_gpu: bool = False):
        super().__init__()
        self._has_vgg = False
        try:
            from torchvision.models import vgg16
            try:
                vgg = vgg16(weights=None)
            except Exception:
                vgg = vgg16(pretrained=False)
            self.feature_extractor = nn.Sequential(*list(vgg.features)[:10])
            self.feature_extractor.eval()
            for p in self.feature_extractor.parameters():
                p.requires_grad = False
            self._has_vgg = True
        except Exception:
            self._has_vgg = False
        self.l1 = nn.L1Loss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_rgb = sr[:, [2, 1, 0], :, :]
        hr_rgb = hr[:, [2, 1, 0], :, :]

        if self._has_vgg:
            try:
                mean = torch.tensor([0.485, 0.456, 0.406], device=sr.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=sr.device).view(1, 3, 1, 1)
                sr_norm = (torch.clamp(sr_rgb, 0.0, 1.0) - mean) / std
                hr_norm = (torch.clamp(hr_rgb, 0.0, 1.0) - mean) / std

                feat_sr = self.feature_extractor(sr_norm)
                feat_hr = self.feature_extractor(hr_norm)
                perceptual_loss = self.l1(feat_sr, feat_hr)
            except Exception:
                perceptual_loss = self.l1(sr_rgb, hr_rgb)
        else:
            perceptual_loss = self.l1(sr_rgb, hr_rgb)

        nir_loss = self.l1(sr[:, 3:4], hr[:, 3:4])
        return 0.7 * perceptual_loss + 0.3 * nir_loss


class BharatSRCombinedLoss(nn.Module):
    """
    Complete Physics-Constrained Multi-Task Loss:
    L_total = lambda_rec * L1 + lambda_sam * L_SAM + lambda_dc * L_DC + lambda_unc * L_uncertainty
    """

    def __init__(
        self,
        scale_factor: int = 4,
        lambda_rec: float = 1.0,
        lambda_sam: float = 0.1,
        lambda_dc: float = 0.1,
        lambda_unc: float = 0.2,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.lambda_rec = lambda_rec
        self.lambda_sam = lambda_sam
        self.lambda_dc = lambda_dc
        self.lambda_unc = lambda_unc

        self.l1_loss = ReconstructionLoss()
        self.sam_loss = SAMLoss()
        self.dc_loss = SpectralConsistencyLoss(scale_factor)
        self.unc_loss = UncertaintyLoss(mode="gaussian")

    def forward(
        self,
        sr: torch.Tensor,
        log_var: Optional[torch.Tensor],
        hr: torch.Tensor,
        lr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_rec = self.l1_loss(sr, hr)
        l_sam = self.sam_loss(sr, hr)
        l_dc = self.dc_loss(sr, lr)

        if log_var is not None:
            l_unc = self.unc_loss(sr, log_var, hr)
        else:
            l_unc = torch.tensor(0.0, device=sr.device)

        total = (
            self.lambda_rec * l_rec
            + self.lambda_sam * l_sam
            + self.lambda_dc * l_dc
            + self.lambda_unc * l_unc
        )

        loss_dict = {
            "total": float(total.item()),
            "l1": float(l_rec.item()),
            "sam": float(l_sam.item()),
            "dc": float(l_dc.item()),
            "unc": float(l_unc.item()) if log_var is not None else 0.0,
        }
        return total, loss_dict


# ========================
# METRICS
# ========================

def compute_psnr(sr, hr, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio.
    Higher is better. Typical good values for SR: 28-35 dB.

    Args:
        sr, hr: numpy arrays (C, H, W) or (H, W)
        max_val: maximum possible value (1.0 for reflectance)
    """
    mse = np.mean((sr.astype(np.float64) - hr.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10 * math.log10(max_val ** 2 / mse)


SENTINEL2_BAND_WEIGHTS = [0.20, 0.25, 0.25, 0.30]  # B2, B3, B4, B8


def compute_ssim(sr, hr, win_size=7, band_weights=None):
    """
    Structural Similarity Index.
    Higher is better. Range [0, 1]. Good SR: > 0.8.
    For multi-band Sentinel-2 imagery, applies band-weighted average (B2: 0.20, B3: 0.25, B4: 0.25, B8: 0.30).
    """
    if sr.ndim == 3:
        if band_weights is None:
            band_weights = SENTINEL2_BAND_WEIGHTS
        c = sr.shape[0]
        weights = band_weights[:c]
        total_w = sum(weights)
        band_ssims = [compute_ssim(sr[b], hr[b], win_size) for b in range(c)]
        return float(sum((w / total_w) * s for w, s in zip(weights, band_ssims)))

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    sr = sr.astype(np.float64)
    hr = hr.astype(np.float64)

    # Use a simple uniform window
    kernel_size = win_size

    # Compute means using uniform filter
    from scipy.ndimage import uniform_filter
    mu_sr = uniform_filter(sr, size=kernel_size)
    mu_hr = uniform_filter(hr, size=kernel_size)

    mu_sr_sq = mu_sr ** 2
    mu_hr_sq = mu_hr ** 2
    mu_sr_hr = mu_sr * mu_hr

    sigma_sr_sq = uniform_filter(sr ** 2, size=kernel_size) - mu_sr_sq
    sigma_hr_sq = uniform_filter(hr ** 2, size=kernel_size) - mu_hr_sq
    sigma_sr_hr = uniform_filter(sr * hr, size=kernel_size) - mu_sr_hr

    ssim_map = ((2 * mu_sr_hr + C1) * (2 * sigma_sr_hr + C2)) / \
               ((mu_sr_sq + mu_hr_sq + C1) * (sigma_sr_sq + sigma_hr_sq + C2))

    return float(np.mean(ssim_map))


def compute_sam(sr, hr):
    """
    Spectral Angle Mapper (SAM).
    Measures the spectral angle between predicted and reference spectra.
    Lower is better. Units: degrees. Good SR: < 5°.

    This metric distinguishes "looks sharp" from "is spectrally correct."

    Args:
        sr, hr: numpy arrays (C, H, W)
    Returns:
        Mean SAM in degrees across all pixels.
    """
    sr = sr.astype(np.float64)
    hr = hr.astype(np.float64)

    # Compute per-pixel spectral angle
    # dot product along spectral dimension
    dot = np.sum(sr * hr, axis=0)
    norm_sr = np.sqrt(np.sum(sr ** 2, axis=0))
    norm_hr = np.sqrt(np.sum(hr ** 2, axis=0))

    # Avoid division by zero
    denom = norm_sr * norm_hr
    denom = np.maximum(denom, 1e-10)

    cos_angle = np.clip(dot / denom, -1, 1)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(np.mean(angle_deg))


def compute_downsample_consistency(sr, lr_original, scale_factor=4):
    """
    Downsample-consistency error.
    Downsample the SR output and compare pixel-wise against the original LR input.

    This is the key explainability metric: if SR is consistent,
    downsampling it should closely reproduce the original input.

    Args:
        sr: (C, H_hr, W_hr) super-resolved output
        lr_original: (C, H_lr, W_lr) original LR input
    Returns:
        Mean absolute error between downsampled SR and original LR.
    """
    c, h_hr, w_hr = sr.shape
    _, h_lr, w_lr = lr_original.shape

    h_trim = min(h_lr * scale_factor, h_hr)
    w_trim = min(w_lr * scale_factor, w_hr)
    sr_trimmed = sr[:, :h_trim, :w_trim]

    h_out = h_trim // scale_factor
    w_out = w_trim // scale_factor

    sr_down = sr_trimmed[:, :h_out * scale_factor, :w_out * scale_factor].reshape(
        c, h_out, scale_factor, w_out, scale_factor
    ).mean(axis=(2, 4))

    # Match target LR shape if needed
    if sr_down.shape != lr_original.shape:
        pad_h = lr_original.shape[1] - sr_down.shape[1]
        pad_w = lr_original.shape[2] - sr_down.shape[2]
        if pad_h > 0 or pad_w > 0:
            sr_down = np.pad(sr_down, ((0, 0), (0, pad_h), (0, pad_w)), mode='edge')

    mae = np.mean(np.abs(sr_down - lr_original))
    return float(mae), sr_down


def compute_spectral_mae(sr: np.ndarray, hr: np.ndarray) -> float:
    """Mean Absolute Error across all spectral bands in reflectance scale."""
    return float(np.mean(np.abs(sr.astype(np.float64) - hr.astype(np.float64))))


def compute_gradient_similarity(sr: np.ndarray, hr: np.ndarray) -> float:
    """
    Spatial gradient and edge similarity using finite-difference Sobel approximation.
    Measures edge direction and sharpness agreement in [0, 1].
    """
    sr_gray = np.mean(sr, axis=0) if sr.ndim == 3 else sr
    hr_gray = np.mean(hr, axis=0) if hr.ndim == 3 else hr

    gx_sr, gy_sr = np.gradient(sr_gray)
    gx_hr, gy_hr = np.gradient(hr_gray)

    mag_sr = np.sqrt(gx_sr ** 2 + gy_sr ** 2)
    mag_hr = np.sqrt(gx_hr ** 2 + gy_hr ** 2)

    c1 = 0.01 ** 2
    edge_sim = (2 * mag_sr * mag_hr + c1) / (mag_sr ** 2 + mag_hr ** 2 + c1)
    return float(np.mean(edge_sim))


def compute_hallucination_and_correctness(
    sr: np.ndarray,
    hr: np.ndarray,
    bicubic: np.ndarray = None,
    edge_threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Quantitative Edge Diagnostics & Correctness Assessment for Remote-Sensing SR.
    Measures:
    - false_edge_rate: Edges generated in SR that do NOT exist in HR.
    - missing_edge_rate: High-frequency edges in HR that SR failed to resolve.
    - high_frequency_excess_rate: Ratio of spurious high-frequency energy beyond HR.
    - high_freq_hallucination_rate: (Legacy alias for high_frequency_excess_rate).
    - consistency_score: Observation consistency based on canonical 4x area degradation.
    - correctness_score: Overall spatial-spectral truthfulness score [0, 1].
    - synthesis_score: Extent of genuine fine-scale texture reconstruction.
    """
    sr_gray = np.mean(sr, axis=0) if sr.ndim == 3 else sr
    hr_gray = np.mean(hr, axis=0) if hr.ndim == 3 else hr

    # Compute spatial gradient magnitudes
    gx_sr, gy_sr = np.gradient(sr_gray)
    gx_hr, gy_hr = np.gradient(hr_gray)
    mag_sr = np.sqrt(gx_sr ** 2 + gy_sr ** 2)
    mag_hr = np.sqrt(gx_hr ** 2 + gy_hr ** 2)

    # Binary edge maps at threshold
    edges_sr = mag_sr > edge_threshold
    edges_hr = mag_hr > edge_threshold

    # False edges: in SR but not HR
    false_edges = edges_sr & (~edges_hr)
    # Missing edges: in HR but not SR
    missing_edges = edges_hr & (~edges_sr)

    hr_edge_count = float(np.sum(edges_hr)) + 1e-7
    sr_edge_count = float(np.sum(edges_sr)) + 1e-7

    false_edge_rate = float(np.sum(false_edges) / sr_edge_count)
    missing_edge_rate = float(np.sum(missing_edges) / hr_edge_count)

    # High frequency residual difference
    hf_diff = np.maximum(0.0, mag_sr - mag_hr)
    hf_excess_rate = float(np.mean(hf_diff) / (np.mean(mag_hr) + 1e-7))

    # Consistency score from Downsample Consistency MAE
    dc_mae, _ = compute_downsample_consistency(sr, degrade_canonical_4x(hr))
    consistency_score = float(max(0.0, min(1.0, 1.0 - (dc_mae / 0.05))))

    # Correctness score combining gradient similarity and 1 - false_edge_rate
    grad_sim = compute_gradient_similarity(sr, hr)
    correctness_score = float(np.clip(0.6 * grad_sim + 0.4 * (1.0 - false_edge_rate), 0.0, 1.0))

    # Synthesis score: real resolved edge detail compared to bicubic baseline
    if bicubic is not None:
        if bicubic.shape[-2:] != sr.shape[-2:]:
            from scipy.ndimage import zoom
            zh = sr.shape[-2] / bicubic.shape[-2]
            zw = sr.shape[-1] / bicubic.shape[-1]
            if bicubic.ndim == 3:
                bic_interp = np.stack([zoom(bicubic[b], (zh, zw), order=3) for b in range(bicubic.shape[0])], axis=0)
            else:
                bic_interp = zoom(bicubic, (zh, zw), order=3)
        else:
            bic_interp = bicubic

        bic_gray = np.mean(bic_interp, axis=0) if bic_interp.ndim == 3 else bic_interp
        gx_b, gy_b = np.gradient(bic_gray)
        mag_bic = np.sqrt(gx_b ** 2 + gy_b ** 2)
        added_hf = np.mean(np.abs(mag_sr - mag_bic))
        target_hf = np.mean(np.abs(mag_hr - mag_bic)) + 1e-7
        synthesis_score = float(np.clip(added_hf / target_hf, 0.0, 1.5))
    else:
        synthesis_score = float(np.clip(np.mean(mag_sr) / (np.mean(mag_hr) + 1e-7), 0.0, 1.5))

    return {
        "false_edge_rate": round(false_edge_rate, 4),
        "missing_edge_rate": round(missing_edge_rate, 4),
        "high_frequency_excess_rate": round(hf_excess_rate, 4),
        "high_freq_hallucination_rate": round(hf_excess_rate, 4),
        "hallucination_rate": round(hf_excess_rate, 4),
        "consistency_score": round(consistency_score, 4),
        "correctness_score": round(correctness_score, 4),
        "synthesis_score": round(synthesis_score, 4),
    }


def compute_all_metrics(sr, hr, lr_original=None, scale_factor=4):
    """
    Compute full scientific remote-sensing metric suite.

    Args:
        sr: (C, H, W) super-resolved output
        hr: (C, H, W) high-resolution reference
        lr_original: (C, H_lr, W_lr) original LR input (optional)
    Returns:
        dict of metric name -> value
    """
    metrics = {
        "psnr_db": round(compute_psnr(sr, hr), 2),
        "ssim": round(compute_ssim(sr, hr), 4),
        "sam_degrees": round(compute_sam(sr, hr), 2),
        "spectral_mae": round(compute_spectral_mae(sr, hr), 6),
        "gradient_similarity": round(compute_gradient_similarity(sr, hr), 4),
    }

    if lr_original is not None:
        dc_error, _ = compute_downsample_consistency(sr, lr_original, scale_factor)
        metrics["downsample_consistency_mae"] = round(dc_error, 6)

    # Edge-based diagnostics & correctness suite
    edge_dict = compute_hallucination_and_correctness(sr, hr)
    metrics.update(edge_dict)

    return metrics
