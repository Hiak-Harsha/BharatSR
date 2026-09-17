# BharatSR — Deep Learning Super-Resolution Mapping for Medium-Resolution Satellite Imagery

> **Smart India Hackathon (SIH 2024 / 2025)**  
> **Problem Statement ID:** SIH26142  
> **Organization:** National Technical Research Organisation (NTRO)  
> **Domain:** Space Technology / Remote Sensing / Defense Analytics  

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI_0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js_16-000000?logo=next.js)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch_2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![Rasterio](https://img.shields.io/badge/GIS-Rasterio_GeoTIFF-green?logo=geopandas)](https://rasterio.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 1. Problem Statement & Motivation

Medium-resolution Earth observation satellites such as **Sentinel-2 (10m–60m)** and **Landsat (15m–30m)** provide high-cadence, multi-spectral global coverage, but are spatially constrained when monitoring micro-land-cover boundaries, military assets, informal settlements, or disaster damage. High-resolution imagery (WorldView, Pleiades) offers sub-meter clarity but is prohibitively expensive and has narrow swath coverage.

**BharatSR** bridges this operational gap by applying **physics-constrained deep learning** to super-resolve 4-band medium-resolution imagery (Red, Green, Blue, Near-Infrared) by **4x spatial factor (10m $\to$ 2.5m GSD)**.

### Why Standard Super-Resolution Fails for Satellite Imagery
Most generic SR models (Bicubic, SRGAN, ESRGAN) treat super-resolution as a cosmetic visual sharpening task:
- ❌ **They normalize using ImageNet mean/std**, destroying physical surface reflectance $[0, 1]$.
- ❌ **They introduce spectral distortion (hallucinated colors)**, breaking quantitative indices like NDVI and mineral mapping.
- ❌ **They fail downsample consistency**, meaning the reconstructed image does not degrade back to the real sensor measurement.
- ❌ **They produce confident hallucinations**, which is dangerous for defense and intelligence photo-interpretation.

### The BharatSR Solution
BharatSR formulates super-resolution as a **physics-constrained spatial regression with spatial uncertainty estimation**:
1. **Reflectance Preservation:** Operates strictly on calibrated top-of-atmosphere (TOA) / bottom-of-atmosphere (BOA) surface reflectance $[0, \sim 1+]$. Bright pixels (clouds, sand, snow $> 1.0$) are never artificially clipped.
2. **Spectral Angle Mapper (SAM):** Enforces angular alignment between multi-spectral bands ($\text{SAM} < 5.0^\circ$), preserving radiometric fidelity.
3. **Vectorized Downsample Consistency:** Ensures that local area-averaged downsampling of the super-resolved output mathematically matches the original sensor input.
4. **Dual-Head Spatial Uncertainty:** Predicts a calibrated heteroscedastic log-variance ($\sigma$) map alongside reflectance, flagging high-frequency textures and potential hallucinations.
5. **Multi-Spectral Analytical Inspection:** Interactive switching between True Color (RGB), False Color Infrared (CIR: NIR-R-G), and Normalized Difference Vegetation Index (NDVI).
6. **Defense & GIS Export:** Exports calibrated 4-band Float32 GeoTIFFs compatible with QGIS, ArcGIS, and GDAL.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph Data_Pipeline [Data Ingestion & Calibration]
        LR_Input[Sentinel-2 / Landsat Tile\n10m GSD - 4 Bands: R, G, B, NIR] --> Preproc[Reflectance Scaling\n[0, ~1+] Range Preserved]
        Preproc --> Cache[Sample Store / Upload Stream]
    end

    subgraph Neural_Models [Deep Learning Super-Resolution Engine]
        Cache --> Bicubic[Bicubic Baseline]
        Cache --> SRCNN[SRCNN Baseline\n3-Layer CNN]
        Cache --> RCAN[Dual-Head RCAN\nResidual-in-Residual + Channel Attention]
        
        RCAN --> Refl_Head[Reflectance Reconstruction Head\n4x Spatial Upscaling: 2.5m GSD]
        RCAN --> Uncert_Head[Uncertainty Head\nSpatial Log-Variance σ Map]
    end

    subgraph Verification [Physics Verification Layer]
        Refl_Head --> SAM_Metric[Spectral Angle Mapper\nSAM < 5°]
        Refl_Head --> DC_Metric[Downsample Consistency\nMAE Degradation Check]
        Refl_Head --> PSNR_SSIM[PSNR & SSIM Evaluation]
        Uncert_Head --> Magma[Magma Heatmap Generator]
    end

    subgraph User_Interface [Mission Control Dashboard]
        SAM_Metric & DC_Metric & PSNR_SSIM & Magma --> FastAPI[FastAPI Backend\nAsync Queue & Model Registry]
        FastAPI --> NextJS[Next.js 16 Web Dashboard]
        NextJS --> MultiSpec[Multi-Spectral Inspector\nRGB / CIR / NDVI / Single Bands]
        NextJS --> Matrix[Multi-Model Benchmark Matrix]
        NextJS --> Slider[Interactive Split Slider]
        NextJS --> GeoExport[4-Band GeoTIFF / JSON Report Export]
    end
```

---

## 3. Model Architectures & Loss Functions

### 1. SRCNN Baseline
A 4-channel adaptation of Dong et al., featuring 3 convolutional stages ($9\times9 \to 1\times1 \to 5\times5$) with 26,084 parameters trained with physical L1 reflectance loss.

### 2. Dual-Head RCAN (Production Architecture)
Residual Channel Attention Network (Zhang et al.) adapted for multi-spectral earth observation:
- **Residual-in-Residual (RIR):** 3 Residual Groups (RG), each containing 3 Residual Channel Attention Blocks (RCAB).
- **Channel Attention Mechanism:** Adaptive inter-band feature recalibration using global average pooling and sigmoid gating.
- **Dual-Head Output:**
  - **Reflectance Head:** Reconstructs 4-band surface reflectance at 4x resolution ($H \times W \to 4H \times 4W$).
  - **Uncertainty Head:** Reconstructs spatial log-variance $s(x, y) = \log(\sigma^2(x, y))$.

### 3. Multi-Task Physics Loss Function
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NLL}}(y, \hat{y}, s) + \lambda_{\text{SAM}} \cdot \mathcal{L}_{\text{SAM}}(y, \hat{y}) + \lambda_{\text{DC}} \cdot \mathcal{L}_{\text{DC}}(\hat{y}, x_{\text{LR}})$$

Where:
- **Heteroscedastic Negative Log-Likelihood (NLL):**
  $$\mathcal{L}_{\text{NLL}} = \frac{1}{2} \exp(-s) \|y - \hat{y}\|_1 + \frac{1}{2} s$$
- **Spectral Angle Mapper (SAM):**
  $$\text{SAM}(y, \hat{y}) = \arccos\left(\frac{\langle y, \hat{y} \rangle}{\|y\|_2 \|\hat{y}\|_2}\right)$$
- **Downsample Consistency (DC):**
  $$\mathcal{L}_{\text{DC}} = \|\mathcal{D}_{\downarrow 4}(\hat{y}) - x_{\text{LR}}\|_1$$

---

## 4. Benchmark & Performance Matrix

Results on calibrated Sentinel-2 validation tiles ($64\times64$ LR $\to$ $256\times256$ HR):

| Metric | Bicubic Baseline | SRCNN Baseline | RCAN Attention (BharatSR) | Target / Spec |
| :--- | :---: | :---: | :---: | :---: |
| **Spatial Factor** | 4x (10m $\to$ 2.5m) | 4x (10m $\to$ 2.5m) | **4x (10m $\to$ 2.5m)** | 4x GSD Enhancement |
| **PSNR (dB)** | 33.72 | 28.68 | **30.12** | $> 28$ dB |
| **SSIM** | 0.782 | 0.724 | **0.812** | $> 0.80$ |
| **SAM (Spectral Angle)** | 3.59° | 4.51° | **3.82°** | $< 5.0^\circ$ (Strict Spectral Fidelity) |
| **Downsample MAE** | 0.0017 | 0.0126 | **0.0094** | $< 0.02$ Physical Preservation |
| **Spatial Uncertainty** | None | None | **Calibrated σ Heatmap** | Defense Hallucination Guard |
| **CPU Latency** | 5 ms | 70 ms | **82 ms** | Real-time interactive ($< 200$ ms) |

---

## 5. Repository Structure

```
SIH26142/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI application with compare, async, & export endpoints
│   │   ├── models_ml/
│   │   │   ├── srcnn.py             # 4-band SRCNN baseline model
│   │   │   ├── rcan.py              # Dual-head RCAN with Channel Attention
│   │   │   └── uncertainty.py       # Log-variance calibration & Magma heatmap generator
│   │   └── services/
│   │       ├── inference.py         # ModelRegistry & inference execution pipeline
│   │       ├── preprocessing.py     # Reflectance normalization, multi-spectral views, GeoTIFF
│   │       ├── postprocessing.py    # PSNR, SSIM, SAM, Downsample MAE computation
│   │       └── job_store.py         # SQLite async job tracking engine
│   ├── sample_tiles/                # Bundled 4-band Sentinel-2 sample tiles
│   ├── weights/                     # Trained checkpoints (srcnn_best.pth, rcan_best.pth)
│   └── tests/
│       ├── test_api_endpoints.py    # Unit tests for baseline endpoints
│       └── test_phase6_endpoints.py # Integration test suite for Phase 6-7 features
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx             # Main dashboard (controls, metrics, benchmark matrix)
│   │   │   └── layout.tsx           # Dark cyber-defense UI layout & fonts
│   │   ├── components/
│   │   │   └── ImageComparisonSlider.tsx # Interactive split slider with spectral switcher
│   │   └── lib/
│   │       └── api.ts               # TypeScript API client & type definitions
│   └── package.json
├── data/
│   └── scripts/
│       ├── prepare_data.py          # Data ingestion pipeline with synthetic fallback
│       └── visualize_patches.py     # Radiometric sanity check script
├── training/
│   ├── losses.py                    # Vectorized physics losses (L1, PSNR, SSIM, SAM, DC MAE)
│   ├── train_srcnn.py               # SRCNN baseline training routine
│   └── train_rcan.py                # Dual-head RCAN physics-constrained training routine
├── run.py                           # Cross-platform application launcher
├── start.bat                        # Windows 1-click launcher
└── README.md
```

---

## 6. Quick Start Guide

### Prerequisites
- Python 3.10+ (with virtual environment in `./venv`)
- Node.js 18+ and npm
- Windows / Linux / macOS

### Option A: Single-Click Launch (Recommended)
On Windows, simply double-click or run:
```cmd
start.bat
```
Or use the Python unified launcher:
```bash
python run.py
```
This automatically verifies checkpoints, boots the FastAPI backend on port `8000`, starts the Next.js frontend on port `3000`, and opens your browser to `http://localhost:3000`.

---

### Option B: Manual Setup

#### 1. Setup Backend
```bash
# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# (Optional) Retrain models
python training/train_srcnn.py
python training/train_rcan.py

# Run FastAPI server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

#### 2. Setup Frontend
```bash
cd frontend
npm install
npm run build
npm run start -- -p 3000
```

---

## 7. API Reference

The FastAPI backend provides OpenAPI documentation at `http://127.0.0.1:8000/docs`.

### Key Endpoints:
- `GET /api/health` — Service health & count of loaded neural models.
- `GET /api/models` — List model architectures and metadata.
- `GET /api/samples` — List pre-loaded multi-band satellite tiles with thumbnails & channel views.
- `POST /api/superresolve` — Execute 4x super-resolution on an image or sample tile.
- `POST /api/compare` — Benchmark Bicubic, SRCNN, and RCAN side-by-side with comparison table.
- `POST /api/superresolve/async` — Submit asynchronous batch job to SQLite queue.
- `GET /api/jobs/{job_id}` — Poll job status and fetch completed result.
- `GET /api/export/geotiff` — Download calibrated 4-band Float32 GeoTIFF.
- `GET /api/export/report` — Download comprehensive analytical evaluation report JSON.

---

## 8. Alignment with NTRO Problem Statement (SIH26142)

| Requirement from Problem Statement | BharatSR Implementation | Evidence |
| :--- | :--- | :--- |
| **Medium-Resolution Input** | Native 4-band Sentinel-2 / Landsat tiles ($10\text{m} \to 2.5\text{m}$) | Multi-band $[C, H, W]$ tensor pipeline |
| **Spatial & Spectral Accuracy** | Spectral Angle Mapper loss + physical reflectance normalization | $\text{SAM} = 3.82^\circ$ ($< 5^\circ$ threshold) |
| **Physics Consistency** | Vectorized Downsample Consistency degradation check | $\text{MAE} < 0.01$ |
| **Avoid Hallucination** | Dual-Head Heteroscedastic Uncertainty Network | Calibrated $\sigma$ spatial heatmap |
| **Micro-Land-Cover Analysis** | NDVI & Color Infrared (CIR) inspection modes | Real-time channel switching in web viewer |
| **Deployment Readiness** | Single-click launcher, CPU inference ($< 100$ ms), GeoTIFF export | Complete FastAPI + Next.js package |

---

## 9. License & Team

Developed for **Smart India Hackathon (SIH 2024 / 2025)** under Problem Statement **SIH26142** for the **National Technical Research Organisation (NTRO)**.  
Licensed under the [MIT License](LICENSE).
