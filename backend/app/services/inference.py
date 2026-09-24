"""
BharatSR — Inference Service
Model loading, inference pipeline, and result generation.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from backend.app.models_ml.srcnn import SRCNN
from backend.app.models_ml.rcan import RCAN
from backend.app.models_ml.swinir_sr import SwinIR_SR
from backend.app.models_ml.hat_sr import HAT_SR
from backend.app.models_ml.diffusion_sr import DiffusionSR
from backend.app.models_ml.ensemble_sr import EnsembleSR
from backend.app.models_ml.uncertainty import logvar_to_std


class ModelRegistry:
    """
    Unified Model Registry: Manages loaded models in memory.
    Supports SRCNN, RCAN, SwinIR, HAT, Diffusion, and Ensemble models.
    Auto-detects CUDA / Apple MPS / CPU.
    """

    def __init__(self):
        self._models: Dict[str, torch.nn.Module] = {}
        self._metadata: Dict[str, dict] = {}
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            print(f"ModelRegistry: GPU detected -> {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
            print("ModelRegistry: Apple Silicon MPS detected")
        else:
            self._device = torch.device("cpu")
            print("ModelRegistry: Running on CPU")

    def load_model(self, model_id: str, checkpoint_path: str) -> bool:
        """Load a model from a checkpoint file."""
        path = Path(checkpoint_path)
        if not path.exists():
            print(f"Checkpoint not found: {path}")
            return False

        try:
            checkpoint = torch.load(str(path), map_location=self._device, weights_only=False)
            n_bands = checkpoint.get("n_bands", 4)
            scale_factor = checkpoint.get("scale_factor", 4)

            if model_id == "srcnn":
                model = SRCNN(n_bands=n_bands)
                name = "SRCNN Baseline"
                arch = "3-layer CNN (L1 loss)"
                has_unc = False
            elif model_id == "rcan":
                n_feats = checkpoint.get("n_feats", 36)
                n_resgroups = checkpoint.get("n_resgroups", 3)
                n_resblocks = checkpoint.get("n_resblocks", 3)
                model = RCAN(
                    n_bands=n_bands,
                    n_feats=n_feats,
                    n_resgroups=n_resgroups,
                    n_resblocks=n_resblocks,
                    scale=scale_factor,
                    predict_uncertainty=True,
                )
                name = "RCAN Attention + Uncertainty"
                arch = f"RIR ({n_resgroups} RGs, {n_resblocks} RCABs, Channel Attention)"
                has_unc = True
            elif model_id == "swinir":
                model = SwinIR_SR(
                    n_bands=n_bands,
                    embed_dim=checkpoint.get("embed_dim", 60),
                    depths=checkpoint.get("depths", [6, 6, 6]),
                    num_heads=checkpoint.get("num_heads", [6, 6, 6]),
                    window_size=checkpoint.get("window_size", 4),
                    scale=scale_factor,
                    predict_uncertainty=True,
                )
                name = "SwinIR Transformer"
                arch = "Shifted-Window Self-Attention Transformer"
                has_unc = True
            elif model_id == "hat":
                model = HAT_SR(
                    n_bands=n_bands,
                    embed_dim=checkpoint.get("embed_dim", 48),
                    depths=checkpoint.get("depths", [4, 4, 4]),
                    num_heads=checkpoint.get("num_heads", [4, 4, 4]),
                    window_size=checkpoint.get("window_size", 4),
                    scale=scale_factor,
                    predict_uncertainty=True,
                )
                name = "HAT Hybrid Attention Transformer"
                arch = "Window + Channel Attention Hybrid Transformer"
                has_unc = True
            elif model_id == "diffusion":
                model = DiffusionSR(
                    n_bands=n_bands,
                    scale=scale_factor,
                    num_steps=checkpoint.get("num_steps", 4),
                )
                name = "DiffusionSR (4-Step DDIM)"
                arch = "Conditional Diffusion with UNet Backbone"
                has_unc = True
            else:
                print(f"Unknown model ID: {model_id}")
                return False

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self._device)
            model.eval()

            self._models[model_id] = model
            self._metadata[model_id] = {
                "id": model_id,
                "name": name,
                "architecture": arch,
                "n_bands": n_bands,
                "scale_factor": scale_factor,
                "epoch": checkpoint.get("epoch", "unknown"),
                "val_loss": checkpoint.get("val_loss", None),
                "has_uncertainty": has_unc,
                "status": "loaded",
            }
            print(f"Loaded model '{model_id}' ({name}) from {path}")
            return True

        except Exception as e:
            print(f"Error loading model '{model_id}': {e}")
            return False

    def build_ensemble(self, model_ids: list, weights: list = None) -> Optional[EnsembleSR]:
        """Build and register an inference-time weighted ensemble from loaded models."""
        models = [self.get_model(m_id) for m_id in model_ids]
        if any(m is None for m in models):
            return None
        weights = weights or [1.0 / len(models)] * len(models)
        ensemble = EnsembleSR(models=models, weights=weights, model_names=model_ids)
        ensemble.to(self._device)
        ensemble.eval()
        self._models["ensemble"] = ensemble
        self._metadata["ensemble"] = {
            "id": "ensemble",
            "name": f"Ensemble ({'+'.join(model_ids)})",
            "architecture": "Variance-Pooled Weighted Multi-Model Ensemble",
            "n_bands": ensemble.n_bands,
            "scale_factor": ensemble.scale_factor,
            "has_uncertainty": ensemble.predict_uncertainty,
            "status": "loaded",
        }
        return ensemble

    def hot_reload(self, model_id: str, checkpoint_path: str) -> bool:
        """Hot-reload a model from an updated checkpoint without server restart."""
        if model_id in self._models:
            del self._models[model_id]
        if model_id in self._metadata:
            del self._metadata[model_id]
        return self.load_model(model_id, checkpoint_path)

    def get_model(self, model_id: str) -> Optional[torch.nn.Module]:
        return self._models.get(model_id)

    def get_metadata(self, model_id: str) -> Optional[dict]:
        return self._metadata.get(model_id)

    def list_models(self) -> list:
        return list(self._metadata.values())

    @property
    def device(self):
        return self._device


def run_inference(
    model: torch.nn.Module,
    lr_image: np.ndarray,
    scale_factor: int = 4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """
    Run super-resolution inference on a single LR image.

    Args:
        model: loaded PyTorch model
        lr_image: (C, H, W) numpy array in reflectance scale
        scale_factor: upscaling factor
        device: torch device

    Returns:
        sr_image: (C, H*scale, W*scale) numpy array
        inference_time: seconds taken
        uncertainty_map: (H*scale, W*scale) standard deviation map or None
    """
    model.eval()

    # Prepare input tensor (1, C, H, W)
    lr_tensor = torch.from_numpy(lr_image).unsqueeze(0).to(device)

    uncertainty_map = None
    t0 = time.time()

    with torch.no_grad():
        if isinstance(model, SRCNN):
            lr_up = F.interpolate(lr_tensor, scale_factor=scale_factor, mode="bicubic", align_corners=False)
            sr_tensor = model(lr_up)
        elif isinstance(model, RCAN):
            res = model(lr_tensor)
            if isinstance(res, tuple):
                sr_tensor, log_var = res
                logvar_np = log_var.squeeze(0).squeeze(0).cpu().numpy()
                uncertainty_map = logvar_to_std(logvar_np)
            else:
                sr_tensor = res
        else:
            sr_tensor = model(lr_tensor)

    inference_time = time.time() - t0
    sr_image = sr_tensor.squeeze(0).cpu().numpy()

    return sr_image, inference_time, uncertainty_map


def _apply_flip(arr: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
    """Flip the last two spatial axes of a (H,W) or (C,H,W) array. Self-inverse."""
    out = arr
    if flip_v:
        out = np.flip(out, axis=-2)
    if flip_h:
        out = np.flip(out, axis=-1)
    return np.ascontiguousarray(out)


def run_inference_ensembled(
    model: torch.nn.Module,
    lr_image: np.ndarray,
    scale_factor: int = 4,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """
    Geometric self-ensemble (test-time augmentation), i.e. "quality" mode.

    Runs inference on the identity input plus its horizontal, vertical, and
    horizontal+vertical flips, undoes each flip on the corresponding output,
    and averages. This costs ~4x latency and zero retraining, and typically
    reduces SAM / improves PSNR because the model's errors are not perfectly
    symmetric under these transforms while the true scene is.

    Uncertainty is averaged in variance (sigma^2) space, not log-variance,
    since variance -- not log-variance -- is the physically additive quantity
    across independent estimates of the same pixel.
    """
    model.eval()
    flip_specs = [(False, False), (True, False), (False, True), (True, True)]

    t0 = time.time()
    sr_accum = None
    var_accum = None
    has_unc = False

    for flip_h, flip_v in flip_specs:
        aug_lr = _apply_flip(lr_image, flip_h, flip_v)
        sr_i, _, unc_i = run_inference(model, aug_lr, scale_factor, device)
        sr_i = _apply_flip(sr_i, flip_h, flip_v)  # flips are self-inverse
        sr_i64 = sr_i.astype(np.float64)
        sr_accum = sr_i64 if sr_accum is None else sr_accum + sr_i64

        if unc_i is not None:
            has_unc = True
            unc_i = _apply_flip(unc_i, flip_h, flip_v)
            var_i = unc_i.astype(np.float64) ** 2
            var_accum = var_i if var_accum is None else var_accum + var_i

    n = len(flip_specs)
    sr_image = (sr_accum / n).astype(np.float32)
    uncertainty_map = np.sqrt(var_accum / n).astype(np.float32) if has_unc else None
    inference_time = time.time() - t0
    return sr_image, inference_time, uncertainty_map


def run_bicubic_baseline(
    lr_image: np.ndarray,
    scale_factor: int = 4
) -> Tuple[np.ndarray, float]:
    """
    Run standard bicubic interpolation baseline.
    Returns (C, H*scale, W*scale) numpy array and latency.
    """
    t0 = time.time()
    lr_tensor = torch.from_numpy(lr_image).unsqueeze(0)
    with torch.no_grad():
        sr_tensor = F.interpolate(
            lr_tensor, scale_factor=scale_factor, mode="bicubic", align_corners=False
        )
    inference_time = time.time() - t0
    sr_image = sr_tensor.squeeze(0).numpy()
    return sr_image, inference_time


def create_blend_window(tile_hr_h: int, tile_hr_w: int) -> np.ndarray:
    """Cosine/Hann 2D window for feathering overlapping tile borders."""
    wy = np.hanning(tile_hr_h + 2)[1:-1]
    wx = np.hanning(tile_hr_w + 2)[1:-1]
    window = np.outer(wy, wx).astype(np.float32)
    return np.maximum(window, 1e-4)


def run_tiled_inference(
    model: torch.nn.Module,
    lr_image: np.ndarray,
    scale_factor: int = 4,
    tile_size: int = 64,
    overlap: int = 16,
    device: torch.device = torch.device("cpu"),
    use_ensemble: bool = False,
) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """
    Memory-safe sliding-window super-resolution inference with overlap blending.
    Enables arbitrary large-image processing without RAM exhaustion or edge artifacts.

    use_ensemble=True runs each tile through the geometric self-ensemble
    (run_inference_ensembled) instead of a single forward pass -- higher
    fidelity, ~4x slower. This is the "quality" mode surfaced in the API.
    """
    infer_fn = run_inference_ensembled if use_ensemble else run_inference

    c, h, w = lr_image.shape
    if h <= tile_size and w <= tile_size:
        return infer_fn(model, lr_image, scale_factor, device)

    t0 = time.time()
    h_out, w_out = h * scale_factor, w * scale_factor
    out_sr = np.zeros((c, h_out, w_out), dtype=np.float32)
    weights = np.zeros((1, h_out, w_out), dtype=np.float32)

    has_unc = bool(getattr(model, "predict_uncertainty", False))
    out_var = np.zeros((h_out, w_out), dtype=np.float32) if has_unc else None
    out_unc = None

    step = max(1, tile_size - overlap)
    y_starts = list(range(0, max(1, h - tile_size + 1), step))
    if y_starts[-1] + tile_size < h:
        y_starts.append(h - tile_size)
    x_starts = list(range(0, max(1, w - tile_size + 1), step))
    if x_starts[-1] + tile_size < w:
        x_starts.append(w - tile_size)

    for y0 in y_starts:
        for x0 in x_starts:
            tile_lr = lr_image[:, y0:y0 + tile_size, x0:x0 + tile_size]
            sr_tile, _, unc_tile = infer_fn(model, tile_lr, scale_factor, device)

            th, tw = sr_tile.shape[1], sr_tile.shape[2]
            win = create_blend_window(th, tw)

            y_out0 = y0 * scale_factor
            x_out0 = x0 * scale_factor

            out_sr[:, y_out0:y_out0 + th, x_out0:x_out0 + tw] += sr_tile * win
            weights[:, y_out0:y_out0 + th, x_out0:x_out0 + tw] += win

            if out_var is not None and unc_tile is not None:
                # Accumulate variance (sigma^2), not sigma directly
                out_var[y_out0:y_out0 + th, x_out0:x_out0 + tw] += (unc_tile ** 2) * win

    out_sr /= np.maximum(weights, 1e-7)
    out_sr = np.clip(out_sr, 0.0, None)
    if out_var is not None:
        out_var /= np.maximum(weights[0], 1e-7)
        out_unc = np.sqrt(np.maximum(out_var, 0.0))

    latency = time.time() - t0
    return out_sr, latency, out_unc


def process_geotiff_file(
    input_path: str,
    output_path: str,
    model: torch.nn.Module,
    scale_factor: int = 4,
    tile_size: int = 64,
    overlap: int = 16,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Production end-to-end GeoTIFF super-resolution pipeline:
    Reads rasterio input -> memory-safe tiled inference -> preserves CRS -> writes 4-band Float32 GeoTIFF.
    """
    import rasterio
    from rasterio.transform import Affine

    with rasterio.open(input_path) as src:
        crs = src.crs
        in_transform = src.transform
        nodata = src.nodata
        raw_data = src.read().astype(np.float32)

    # Reflectance scaling
    if raw_data.max() > 10.0:
        raw_data = raw_data / 10000.0
    elif raw_data.max() > 1.5:
        raw_data = raw_data / 255.0
    raw_data = np.clip(raw_data, 0.0, None)

    if raw_data.shape[0] < 4:
        raise ValueError(
            f"Sentinel-2 model requires exactly 4 bands (B2, B3, B4, B8). "
            f"File has {raw_data.shape[0]} bands. "
            "Zero-padding bands is not allowed. Please provide a 4-band GeoTIFF."
        )
    elif raw_data.shape[0] > 4:
        raw_data = raw_data[:4]

    c_in, h_in, w_in = raw_data.shape
    sr_image, latency, unc_map = run_tiled_inference(
        model=model,
        lr_image=raw_data,
        scale_factor=scale_factor,
        tile_size=tile_size,
        overlap=overlap,
        device=device,
    )

    out_transform = Affine(
        in_transform.a / scale_factor,
        in_transform.b,
        in_transform.c,
        in_transform.d,
        in_transform.e / scale_factor,
        in_transform.f,
    )

    c, h, w = sr_image.shape
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=np.float32,
        crs=crs,
        transform=out_transform,
        nodata=nodata,
    ) as dst:
        for b in range(c):
            dst.write(sr_image[b], b + 1)
        band_names = ["B2 - Blue", "B3 - Green", "B4 - Red", "B8 - NIR"]
        for b in range(min(c, len(band_names))):
            dst.set_band_description(b + 1, band_names[b])
        dst.update_tags(
            sensor="Sentinel-2 MSI",
            processing=f"BharatSR {scale_factor}x Super-Resolution",
            gsd=f"{abs(out_transform.a):.2f}m-equivalent output grid",
        )

    return {
        "status": "success",
        "output_path": output_path,
        "crs": str(crs),
        "scale_factor": scale_factor,
        "input_size": [h_in, w_in],
        "output_size": [h, w],
        "dimensions": [c, h, w],
        "latency_seconds": round(latency, 2),
    }


def run_batch_inference(
    model: torch.nn.Module,
    lr_images: List[np.ndarray],  # List of (C, H, W) arrays
    scale_factor: int = 4,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 4,
) -> List[Tuple[np.ndarray, float, Optional[np.ndarray]]]:
    """
    Efficient batched inference for multiple LR tiles.
    Pads images to a common size within each mini-batch.
    Returns list of (sr_image, latency, uncertainty_map) per input.
    """
    results = []
    for i in range(0, len(lr_images), batch_size):
        batch = lr_images[i : i + batch_size]
        max_h = max(img.shape[1] for img in batch)
        max_w = max(img.shape[2] for img in batch)

        padded = []
        for img in batch:
            c, h, w = img.shape
            pad_h, pad_w = max_h - h, max_w - w
            padded.append(np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect"))

        batch_tensor = torch.stack([torch.from_numpy(p) for p in padded]).float().to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model(batch_tensor)
        latency = (time.time() - t0) / max(1, len(batch))

        if isinstance(out, tuple):
            sr_batch, lv_batch = out
        else:
            sr_batch, lv_batch = out, None

        for j, original in enumerate(batch):
            _, h_orig, w_orig = original.shape
            sr = sr_batch[j, :, : h_orig * scale_factor, : w_orig * scale_factor].cpu().numpy()
            unc = None
            if lv_batch is not None:
                unc = logvar_to_std(
                    lv_batch[j, 0, : h_orig * scale_factor, : w_orig * scale_factor].cpu().numpy()
                )
            results.append((sr, latency, unc))
    return results


# Global registry instance
model_registry = ModelRegistry()


