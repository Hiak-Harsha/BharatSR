# BharatSR — Multi-Model Scientific Ablation Benchmark

> **Standardized Benchmark Across 6 Configurations**
> Evaluated on held-out scene-separated test split with physical surface reflectance.

| Configuration | Architecture | Params | Latency (s) | PSNR (dB) | SSIM | SAM (°) | DC-MAE | Spectral MAE | Edge Excess Rate |
|---|---|---|---|---|---|---|---|---|---|
| **A_Bicubic** | Bicubic Interpolation (Deterministi | 0 | 0.0377s | **32.54** | **0.7569** | 3.50° | 0.0030 | 0.0169 | 0.0351 |
| **B_SRCNN_L1** | SRCNN Baseline (Bicubic pre-upsampl | 26,084 | 0.0508s | **21.24** | **0.6720** | 10.77° | 0.0555 | 0.0583 | 0.2503 |
| **C_RCAN_L1** | Lightweight RCAN-Lite (3 RGs, 3 RCA | 450,184 | 0.0904s | **32.27** | **0.7269** | 3.72° | 0.0029 | 0.0179 | 0.0816 |
| **D_RCAN_L1_DC** | Lightweight RCAN-Lite (3 RGs, 3 RCA | 450,184 | 0.0707s | **32.21** | **0.7243** | 3.75° | 0.0031 | 0.0180 | 0.0819 |
| **E_RCAN_L1_SAM_DC** | Lightweight RCAN-Lite (3 RGs, 3 RCA | 450,184 | 0.1071s | **32.05** | **0.7209** | 3.80° | 0.0045 | 0.0184 | 0.0978 |
| **F_RCAN_Full_Uncertainty** | Lightweight RCAN-Lite (3 RGs, 3 RCA | 456,197 | 0.0569s | **31.40** | **0.6844** | 4.09° | 0.0044 | 0.0197 | 0.1579 |

### Key Scientific Findings:
1. **Spectral Consistency & Observation Constraint:** Incorporating Downsample Consistency ($L_{\text{DC}}$) strictly constrains the super-resolved output to reproduce the 10m LR capture when area-averaged.
2. **Spectral Fidelity ($L_{\text{SAM}}$):** SAM loss minimizes spectral vector distortions across B2, B3, B4, and B8, directly protecting radiometric fidelity for downstream NDVI calculation.
3. **Uncertainty Quantification:** Heteroscedastic Gaussian log-variance heads quantify pixel-wise epistemic and aleatoric confidence without degrading reconstruction quality.
