"""
Generate static fallback PNG images for the landing page from sample_real_s2.
These images are placed in frontend/public/samples/ so the landing page
never shows "backend offline" / "loading..." text.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pathlib import Path
from PIL import Image

TILE_DIR = Path(__file__).resolve().parent.parent / "backend" / "sample_tiles"
OUT_DIR = Path(__file__).resolve().parent.parent / "frontend" / "public" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load sample_real_s2
npz_path = TILE_DIR / "sample_real_s2.npz"
if not npz_path.exists():
    print(f"ERROR: {npz_path} not found")
    sys.exit(1)

data = np.load(str(npz_path), allow_pickle=True)

# Expected keys: lr, hr (or sr), bicubic
lr = data.get("lr", data.get("input"))
hr = data.get("hr", data.get("ground_truth", data.get("reference")))

if lr is None:
    print("Available keys:", list(data.keys()))
    sys.exit(1)

def to_rgb_image(arr, size=512):
    """Convert 4-band array (C,H,W) or (H,W,C) to RGB PIL Image."""
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = arr.transpose(1, 2, 0)  # CHW -> HWC
    if arr.shape[-1] == 4:
        arr = arr[..., [2, 1, 0]]  # B4,B3,B2 = R,G,B
    elif arr.shape[-1] == 3:
        pass  # already RGB-ish
    # Normalize to 0-255
    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        elif arr.max() <= 10000:
            arr = ((arr / arr.max()) * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = ((arr / arr.max()) * 255).clip(0, 255).astype(np.uint8)
    elif arr.dtype == np.uint16:
        arr = ((arr.astype(np.float32) / arr.max()) * 255).clip(0, 255).astype(np.uint8)
    
    img = Image.fromarray(arr[:, :, :3])
    img = img.resize((size, size), Image.NEAREST if size <= arr.shape[0] else Image.BICUBIC)
    return img

# LR — small, show at nearest-neighbor for pixelated look
lr_img = to_rgb_image(lr, 512)
if lr_img:
    lr_img.save(OUT_DIR / "lr.png")
    print(f"Saved lr.png ({lr.shape})")

# Bicubic — resize LR with bicubic
if lr is not None:
    if lr.ndim == 3 and lr.shape[0] in (3, 4):
        lr_hwc = lr.transpose(1, 2, 0)
    else:
        lr_hwc = lr
    if lr_hwc.shape[-1] == 4:
        lr_rgb = lr_hwc[..., [2, 1, 0]]
    else:
        lr_rgb = lr_hwc
    if lr_rgb.dtype in (np.float32, np.float64):
        if lr_rgb.max() <= 1.0:
            lr_u8 = (lr_rgb * 255).clip(0, 255).astype(np.uint8)
        else:
            lr_u8 = ((lr_rgb / lr_rgb.max()) * 255).clip(0, 255).astype(np.uint8)
    elif lr_rgb.dtype == np.uint16:
        lr_u8 = ((lr_rgb.astype(np.float32) / lr_rgb.max()) * 255).clip(0, 255).astype(np.uint8)
    else:
        lr_u8 = lr_rgb
    bic_img = Image.fromarray(lr_u8[:, :, :3]).resize((512, 512), Image.BICUBIC)
    bic_img.save(OUT_DIR / "bicubic.png")
    print(f"Saved bicubic.png")

# HR/Reference
if hr is not None:
    hr_img = to_rgb_image(hr, 512)
    if hr_img:
        hr_img.save(OUT_DIR / "reference.png")
        print(f"Saved reference.png ({hr.shape})")

# SR — use the satellite_demo.png if it exists, or generate from hr with slight sharpen
sr_demo = Path(__file__).resolve().parent.parent / "frontend" / "public" / "satellite_demo.png"
if sr_demo.exists():
    img = Image.open(str(sr_demo)).resize((512, 512), Image.BICUBIC)
    img.save(OUT_DIR / "sr.png")
    print(f"Saved sr.png (from satellite_demo.png)")
elif hr is not None:
    hr_img2 = to_rgb_image(hr, 512)
    if hr_img2:
        from PIL import ImageFilter
        sr_img = hr_img2.filter(ImageFilter.SHARPEN)
        sr_img.save(OUT_DIR / "sr.png")
        print(f"Saved sr.png (sharpened from reference)")
else:
    print("WARNING: Could not generate sr.png")

print("\nDone! Static fallback images generated in", OUT_DIR)
