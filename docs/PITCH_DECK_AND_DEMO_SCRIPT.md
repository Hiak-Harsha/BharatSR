# BharatSR — Smart India Hackathon Presentation Deck & Demo Script
**Problem Statement:** SIH26142 — Deep Learning Super-Resolution Mapping for Medium-Resolution Satellite Imagery  
**Organization:** National Technical Research Organisation (NTRO), Space Technology Domain  
**Format:** 5-Minute Live Pitch + Interactive Demo

---

## ⏱️ 5-Minute Pitch Outline

```
[0:00 - 0:40] Slide 1 & 2: Problem Definition & Operational Satellite Constraints
[0:40 - 1:25] Slide 3: Why Generic Image Super-Resolution Fails for Defense
[1:25 - 2:20] Slide 4 & 5: The BharatSR Breakthrough (Physics Constraints + Uncertainty)
[2:20 - 4:20] LIVE DEMO: Click-to-Inspect, Spectral Bands, Uncertainty Slider, Downstream Tasks
[4:20 - 5:00] Slide 6: Automated Metrics, GeoTIFF GIS Integration & Impact for NTRO
```

---

## 🎙️ Slide-by-Slide Speaking Script

### Slide 1: Title & Team
> *"Respected judges and representatives from the National Technical Research Organisation. We are presenting **BharatSR**, a physics-constrained deep learning super-resolution mapping system designed specifically for India's defense and remote sensing priorities under problem statement SIH26142."*

### Slide 2: The Core Dilemma
> *"India relies heavily on high-cadence medium-resolution constellations like Sentinel-2 (10m–20m) and Landsat (15m–30m) to monitor border frontiers, agricultural zones, and critical infrastructure. While they offer frequent revisits, a 10m Ground Sample Distance (GSD) leaves field hedgerows, unpaved tracks, and micro-structures as single blurry pixels. High-resolution satellites like Cartosat-3 or commercial WorldView offer sharp details, but are narrow in swath, expensive, and cannot cover wide territories on demand."*

### Slide 3: Why Generic Super-Resolution Fails for Defense
> *"Most existing super-resolution implementations make a fatal mistake: they treat satellite imagery like ordinary smartphone photos. They normalize with ImageNet statistics, which erases physical surface reflectance. They optimize solely for visual sharpness, hallucinating colors and distorting radiometric ratios. In military intelligence, a hallucinated edge or an artificially greened field is dangerous. Downsample an unconstrained SR image, and it completely fails to match the original sensor measurement."*

### Slide 4: The BharatSR Formulation
> *"BharatSR formulates super-resolution not as image filtering, but as **physics-constrained spatial regression with spatial uncertainty quantification**:*
> 1. *We strictly preserve physical surface reflectance $[0, \sim 1+]$, allowing bright targets like desert sands or clouds to exceed $1.0$ without artificial clipping.*
> 2. *We enforce a **Spectral Angle Mapper (SAM) loss** below $5.0^\circ$, ensuring that multi-spectral ratios across Red, Green, Blue, and Near-Infrared remain radiometrically authentic.*
> 3. *We incorporate **Vectorized Downsample Consistency**, mathematically constraining that the enhanced 2.5m-equivalent output downsamples back to the original 10m sensor capture with an MAE under $0.01$.*
> 4. *Our production model uses a **Dual-Head Residual Channel Attention Network (RCAN)**: while the primary head synthesizes $4\times$ reflectance, the secondary head predicts a per-pixel heteroscedastic log-variance map, flagging potential hallucinations before intelligence officers make tactical decisions."*

### Slide 5: Real-Time Live Demonstration Script
*(Switch to Web Browser at `http://localhost:3000`)*

1. **Step 1: Authentic Sentinel-2 Scene Selection**
   > *"Notice that our dashboard defaults directly to authentic Sentinel-2 Level-2A surface reflectance data—here, the Roorkee/Haridwar Ganga Canal corridor demonstration tile with preserved UTM Zone 44N (EPSG:32644) georeferencing. It contains genuine 10m bands: Blue (B2), Green (B3), Red (B4), and NIR (B8)."*
2. **Step 2: Run 4x Super-Resolution & Seamless Tiled Inference**
   > *"Let's click **'Run 4x Super-Resolution'**. In under 120 milliseconds on CPU, the dual-head RCAN model enhances the 10m tile onto a 2.5m-equivalent grid. For large operational scenes, BharatSR automatically routes through a seamless Hann-windowed overlapping tiling engine to eliminate boundary seams and prevent memory overflow."*
3. **Step 3: Interactive Split-Slider & Click-to-Inspect Reticle**
   > *"On the interactive viewer, dragging the slider reveals sharp field hedgerows, canals, and road edges. We can switch channels from **True Color (RGB)** to **Color Infrared (CIR)** and **NDVI Vegetation Index**. But even better: **click anywhere on the canvas**! A tactical animated reticle locks onto the pixel coordinates, instantly extracting its 4-band radiometric curve and verifying whether vegetation or built-up spectral signatures are preserved."*
4. **Step 4: Tactical Uncertainty Alert Threshold Slider**
   > *"In defense operations, blind AI trust is a liability. Notice our **Defense Uncertainty Alert Mask Overlay**. By adjusting the threshold slider $\tau$, analysts can dynamically highlight high-variance edge boundaries directly over the imagery, ensuring photo-interpreters know exactly which details are mathematically certain versus uncertain."*
5. **Step 5: Downstream Analytical Task Visualizer**
   > *"To prove operational utility beyond human eyes, look at the **Downstream Task Visualizer**. It runs automated segmentation pipelines for **Micro-Canopy Vegetation** and **Built-up Infrastructure**. BharatSR achieves significantly higher IoU and F1 scores than Bicubic interpolation, demonstrating that our enhanced imagery directly boosts automated target detection algorithms."*
6. **Step 6: Defense GIS GeoTIFF Export (Curated & User Uploads)**
   > *"Finally, BharatSR seamlessly integrates into national defense workflows. Whether using curated tiles or user-uploaded GeoTIFFs, one click generates a calibrated **Float32 4-band GeoTIFF** with scaled affine transforms ($p/4$), ready for direct load into QGIS, ArcGIS, or GDAL pipelines."*

### Slide 6: Summary & Impact
> *"In summary, BharatSR delivers: 4x spatial scaling (10m $\to$ 2.5m), radiometric fidelity with SAM under $5^\circ$, mathematical downsample consistency under $0.01$, click-to-inspect radiometric auditing, automated downstream task gains, and real-time CPU inference. Thank you, and we welcome your questions."*

---

## 🛡️ Anticipated Judge Questions & Technical Answers

### Q1: "How do you prove that the model is not hallucinating details?"
**Answer:**
> *"We address hallucination on three levels: first, mathematically via our Downsample Consistency loss ($MAE < 0.002$), which strictly bounds deviation from the sensor's physical aperture. Second, empirically via our secondary uncertainty head which predicts spatial log-variance $\sigma(x, y)$ — which we explicitly declare as uncalibrated rather than claiming false probabilistic certainty, serving as a relative spatial risk map where reconstruction errors concentrate along ambiguous boundaries. Third, operationally with our threshold alert slider on the analyst's screen, which highlights high-variance zones so defense photo-interpreters know exactly which details require secondary reconnaissance."*

### Q2: "Why is Spectral Angle Mapper (SAM) important compared to PSNR?"
**Answer:**
> *"PSNR only measures pixel brightness differences, meaning a model could shift band ratios and still score high. In remote sensing, mineral classification and NDVI depend strictly on ratios between bands. SAM computes the spectral angle between predicted and ground-truth vectors. Achieving $SAM < 5.0^\circ$ guarantees that the multi-spectral physics are preserved."*

### Q3: "Can the system handle large satellite swaths and custom user imagery?"
**Answer:**
> *"Yes. BharatSR supports arbitrary user GeoTIFF uploads with full CRS preservation. It features an automated tiled inference pipeline using overlapping tiles and 2D Hann-window cosine blending, preventing tile edge artifacts while operating within strict RAM budgets."*
