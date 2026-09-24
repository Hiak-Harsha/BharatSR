"""
BharatSR — Machine Learning Model Zoo
"""

from .srcnn import SRCNN
from .rcan import RCAN
from .swinir_sr import SwinIR_SR
from .hat_sr import HAT_SR
from .diffusion_sr import DiffusionSR
from .ensemble_sr import EnsembleSR

MODEL_REGISTRY_NAMES = {
    "srcnn": SRCNN,
    "rcan": RCAN,
    "swinir": SwinIR_SR,
    "hat": HAT_SR,
    "diffusion": DiffusionSR,
    "ensemble": EnsembleSR,
}

__all__ = [
    "SRCNN",
    "RCAN",
    "SwinIR_SR",
    "HAT_SR",
    "DiffusionSR",
    "EnsembleSR",
    "MODEL_REGISTRY_NAMES",
]
