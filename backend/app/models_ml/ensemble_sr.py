"""
BharatSR — EnsembleSR (Inference-Time Multi-Model Ensemble)
Combines predictions from complementary architectures (e.g., RCAN + SwinIR)
via calibrated weighted averaging in reflectance space and variance-space uncertainty pooling.
"""

from typing import Tuple, List, Optional, Union
import torch
import torch.nn as nn


import torch.nn.functional as F


class EnsembleSR(nn.Module):
    """
    Weighted ensemble of trained super-resolution models.
    Supports arbitrary combinations (default: 60% RCAN, 40% SwinIR).
    """
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        model_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        else:
            w_sum = sum(weights)
            weights = [w / w_sum for w in weights]
        self.weights = weights
        self.model_names = model_names or [f"model_{i}" for i in range(len(models))]
        self.scale_factor = getattr(models[0], "scale", getattr(models[0], "scale_factor", 4))
        self.n_bands = getattr(models[0], "n_bands", 4)
        self.predict_uncertainty = any(getattr(m, "predict_uncertainty", False) for m in models)

    def forward(self, lr: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        sr_list = []
        var_list = []

        for m, w in zip(self.models, self.weights):
            if m.__class__.__name__ == "SRCNN":
                lr_in = F.interpolate(lr, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
                out = m(lr_in)
            else:
                out = m(lr)
            if isinstance(out, tuple):
                sr, lv = out
                sr_list.append((sr, w))
                var_list.append((torch.exp(lv), w))
            else:
                sr_list.append((out, w))

        # Weighted reflectance average
        sr_ens = sum(sr * w for sr, w in sr_list)

        if var_list:
            # Variance pooling across model predictions that output uncertainty
            total_var_w = sum(w for _, w in var_list)
            var_ens = sum(var * (w / total_var_w) for var, w in var_list)
            log_var_ens = torch.log(var_ens + 1e-7)
            log_var_ens = torch.clamp(log_var_ens, min=-6.0, max=6.0)
            return sr_ens, log_var_ens

        return sr_ens

