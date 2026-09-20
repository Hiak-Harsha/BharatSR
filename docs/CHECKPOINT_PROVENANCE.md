# BharatSR Checkpoint & Training Provenance Manifest

> **Problem Statement ID:** SIH26142 (NTRO Space Technology / Remote Sensing)  
> **Status:** Active & Audited  
> **Policy:** Zero Data Fabrication — Full Reproducibility

---

## 1. Checkpoint Inventory & Integrity Hashes

Every model checkpoint bundled in `backend/weights/` has been cryptographically hashed (SHA-256) and verified:

| Checkpoint File | Architecture | Parameters | File Size | SHA-256 Hash |
| :--- | :--- | :---: | :---: | :--- |
| **`rcan_best.pth`** | RCAN Dual-Head (Residual Anchor + RIR + Uncertainty) | 456,197 | 1,858,645 bytes | `4b63551bd78e15ef2d7ef4b6bd476d38484eb96ad7b276eef29b7c899c1fca45` |
| **`srcnn_best.pth`** | SRCNN Baseline ($9\times9 \to 1\times1 \to 5\times5$) | 26,084 | 107,513 bytes | `fcd4624f14d1d26dc924e99f74e2ced8a3addf0867633bb52afb215a86f4f2c3` |
| **`rcan_ablation_b.pth`** | RCAN-Lite (Config B - SRCNN Checkpoint) | 26,084 | 1,833,857 bytes | `8c7f4afbbfe40eedc6f1720efe8445f96f4efd21880f746459460e420f79e297` |
| **`rcan_ablation_c.pth`** | RCAN-Lite (L1 Loss Only) | 450,184 | 1,833,857 bytes | `566ab7581a5423144e960adeff6766659b2e825de6a08e0a335cd32f73f1a690` |
| **`rcan_ablation_d.pth`** | RCAN-Lite (L1 + Downsample Consistency) | 450,184 | 1,833,857 bytes | `e84ddce1592d9da2466d73c513af0037fb39aae055eb85de72a250e400ee23d9` |
| **`rcan_ablation_e.pth`** | RCAN-Lite (L1 + SAM + Downsample Consistency) | 450,184 | 1,833,857 bytes | `65cafe296e419c31616e5830d3eeb6043f09a85aabf93da3fae334358cce1243` |

---

## 2. Training Dataset Provenance

### Transparent Declaration:
The bundled demonstration checkpoints were trained on the **procedural multi-spectral development dataset** generated via `data/scripts/prepare_data.py --synthetic` using canonical area-averaging sensor degradation.

```text
Dataset:
  Synthetic Multi-Spectral Procedural Dataset (Development Mode)

Number of Scenes:
  200 total scenes generated under seed=42

Scene Partitioning (Zero Spatial Leakage):
  - Training:   140 scenes (scenes 0–139, 840 augmented patches)
  - Validation: 30 scenes  (scenes 140–169, 30 patches)
  - Testing:    30 scenes  (scenes 170–199, 30 patches)

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
- **Validation Loss:** $-0.5538$ (best epoch 5, early stopping patience 3).

### B. Baseline Model (`srcnn_best.pth` / Config B)
- **Architecture:** 4-channel adapted SRCNN ($9\times9 \to 1\times1 \to 5\times5$).
- **Optimizer:** Adam ($\eta = 5 \times 10^{-4}$, 6 epochs).
- **Loss:** Standard L1 reflectance loss.
- **Validation Loss:** $0.0189$ (best epoch 5).

---

## 4. How to Reproduce Training

To reproduce or retrain any checkpoint from scratch:

```bash
# 1. Generate the scene-separated development dataset (200 scenes)
python data/scripts/prepare_data.py --synthetic --n-scenes 200 --seed 42

# 2. Train SRCNN Baseline
python training/train_srcnn.py

# 3. Train RCAN Production Model
python training/train_rcan.py

# 4. Execute the complete 6-configuration scientific ablation suite
python evaluation/run_ablations.py
```
