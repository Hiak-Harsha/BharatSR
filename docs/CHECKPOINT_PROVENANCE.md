# BharatSR Checkpoint & Training Provenance Manifest

> **Problem Statement ID:** SIH26142 (NTRO Space Technology / Remote Sensing)  
> **Status:** Active & Audited  
> **Policy:** Zero Data Fabrication — Full Reproducibility

---

## 1. Checkpoint Inventory & Integrity Hashes

Every model checkpoint bundled in `backend/weights/` has been cryptographically hashed (SHA-256) and verified:

| Checkpoint File | Architecture | Parameters | File Size | SHA-256 Hash |
| :--- | :--- | :---: | :---: | :--- |
| **`rcan_best.pth`** | RCAN Dual-Head (Residual Anchor + RIR + Uncertainty) | 456,197 | 1,858,645 bytes | `d42c7effd66f495ebc71a27932ebd2508cb3b0f61d21b7578676679003b24a1a` |
| **`srcnn_best.pth`** | SRCNN Baseline ($9\times9 \to 1\times1 \to 5\times5$) | 26,084 | 321,013 bytes | `d730e30ac812356dcbfecb522e34374ccb70fb50e8e5d9409d67bf5706b14744` |
| **`rcan_ablation_c.pth`** | RCAN-Lite (L1 Loss Only) | 450,184 | 1,833,857 bytes | `273cfcbc507dd4393c40d00cbfc7dcb55980aa661e00d3217e0aeaef24e65632` |
| **`rcan_ablation_d.pth`** | RCAN-Lite (L1 + Downsample Consistency) | 450,184 | 1,833,857 bytes | `ebbe7aaec1e4d91b84c9fc88e1308112c3d75f6a350c4d2fd873cea2d505fcba` |
| **`rcan_ablation_e.pth`** | RCAN-Lite (L1 + SAM + Downsample Consistency) | 450,184 | 1,833,857 bytes | `2e989a1cb67f7e879e2616bdf01a1344efe795a13cd2c56bd981f94e2abb8b4e` |

---

## 2. Training Dataset Provenance

### Transparent Declaration:
The bundled demonstration checkpoints were trained on the **procedural multi-spectral development dataset** generated via `data/scripts/prepare_data.py --synthetic` using canonical area-averaging sensor degradation.

```text
Dataset:
  Synthetic Multi-Spectral Procedural Dataset (Development Mode)

Number of Scenes:
  24 total scenes generated under seed=42

Scene Partitioning (Zero Spatial Leakage):
  - Training:   16 scenes (scenes 0–15)
  - Validation: 4 scenes  (scenes 16–19)
  - Testing:    4 scenes  (scenes 20–23)

Spectral Bands:
  4 Bands corresponding to Sentinel-2 Level-2A VNIR channels:
  - Band 0: B2 (Blue, 490 nm)
  - Band 1: B3 (Green, 560 nm)
  - Band 2: B4 (Red, 665 nm)
  - Band 3: B8 (Near-Infrared, 842 nm)

Degradation Operator Applied During Training:
  Canonical area-averaging downsampling:
  D(HR) = avg_pool2d(HR, kernel_size=4, stride=4) == LR

Synthetic Data Status:
  EXPLICITLY DECLARED AND DOCUMENTED.
  Demonstration checkpoints verify architectural execution, pipeline integrity,
  geospatial transforms, and physics losses. They have NOT been fine-tuned on
  restricted or proprietary defense satellite acquisitions.
```

---

## 3. Training Hyperparameters & Loss Configuration

### A. Production Model (`rcan_best.pth` / Config F)
- **Architecture:** Residual-in-Residual (3 Residual Groups, 3 RCABs per group, 36 internal features, channel attention reduction ratio 16).
- **Residual Anchor:** $\hat{y} = \text{Bicubic}(x_{\text{LR}}) + \mathcal{F}_{\text{RCAN}}(x_{\text{LR}}; \theta)$
- **Uncertainty Head:** Heteroscedastic Gaussian log-variance $s = \log(\sigma^2)$.
- **Optimizer:** AdamW (learning rate $\eta = 5 \times 10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, weight decay $10^{-4}$).
- **Loss Weights:**
  - $\lambda_{\text{rec}} = 1.0$ (L1 reflectance loss)
  - $\lambda_{\text{sam}} = 0.1$ (Spectral Angle Mapper loss, $\epsilon = 10^{-7}$)
  - $\lambda_{\text{dc}} = 0.1$ (Canonical Downsample Consistency loss)
  - $\lambda_{\text{unc}} = 0.2$ (Gaussian negative log-likelihood with log-variance regularization)
- **Validation Loss:** $-0.0571$ (NLL scale).

### B. Baseline Model (`srcnn_best.pth` / Config B)
- **Architecture:** 4-channel adapted SRCNN ($9\times9 \to 1\times1 \to 5\times5$).
- **Optimizer:** Adam ($\eta = 10^{-3}$, 10 epochs).
- **Loss:** Standard L1 reflectance loss.
- **Validation Loss:** $0.0257$.

---

## 4. How to Reproduce Training

To reproduce or retrain any checkpoint from scratch:

```bash
# 1. Generate the scene-separated development dataset
python data/scripts/prepare_data.py --synthetic --n-scenes 24 --seed 42

# 2. Train SRCNN Baseline
python training/train_srcnn.py

# 3. Train RCAN Production Model
python training/train_rcan.py

# 4. Execute the complete 6-configuration scientific ablation suite
python evaluation/run_ablations.py
```
