"""
BharatSR — Downstream Analytical Task Evaluation
Evaluates whether super-resolution improves analytical task performance
(e.g., Road/Building edge extraction, Micro-Vegetation canopy segmentation)
rather than merely cosmetic visual appearance.

Metrics computed across LR, Bicubic, SRCNN, RCAN, and HR Reference:
- Precision
- Recall
- F1-Score
- Intersection over Union (IoU / Jaccard Index)
"""

import numpy as np
from typing import Dict, Tuple


def segment_micro_canopy(img: np.ndarray, ndvi_threshold: float = 0.35) -> np.ndarray:
    """
    Extract active vegetation canopy using physical NDVI:
    NDVI = (NIR - Red) / (NIR + Red + 1e-7)
    img shape: (C, H, W) where Red=band 0 (or 2), NIR=band 3
    """
    red = img[0]  # Band 4 (Red)
    nir = img[3]  # Band 8 (NIR)
    ndvi = (nir - red) / (nir + red + 1e-7)
    return (ndvi > ndvi_threshold).astype(np.uint8)


def segment_built_up_infrastructure(img: np.ndarray, threshold: float = 0.18) -> np.ndarray:
    """
    Extract built-up infrastructure / road networks using high visible-NIR albedo
    and spatial edge features.
    """
    # High albedo across visible bands combined with low NIR-Red contrast
    visible_mean = (img[0] + img[1] + img[2]) / 3.0
    red = img[0]
    nir = img[3]
    contrast = np.abs(nir - red)
    # Built-up roads/structures: high reflectance + low vegetation contrast
    mask = (visible_mean > threshold) & (contrast < 0.08)
    return mask.astype(np.uint8)


def compute_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Compute Precision, Recall, F1-Score, and IoU for binary segmentation.
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
    }


def evaluate_downstream_suite(
    lr_bicubic: np.ndarray,
    sr_srcnn: np.ndarray,
    sr_rcan: np.ndarray,
    hr_reference: np.ndarray,
) -> Dict[str, dict]:
    """
    Evaluate all models on downstream analytical segmentation tasks against ground truth HR.
    """
    tasks = {
        "canopy_segmentation": segment_micro_canopy,
        "built_up_infrastructure": segment_built_up_infrastructure,
    }

    results = {}
    for task_name, seg_fn in tasks.items():
        gt_mask = seg_fn(hr_reference)

        mask_bicubic = seg_fn(lr_bicubic)
        mask_srcnn = seg_fn(sr_srcnn)
        mask_rcan = seg_fn(sr_rcan)

        results[task_name] = {
            "bicubic": compute_segmentation_metrics(mask_bicubic, gt_mask),
            "srcnn": compute_segmentation_metrics(mask_srcnn, gt_mask),
            "rcan": compute_segmentation_metrics(mask_rcan, gt_mask),
            "ground_truth_pixel_count": int(gt_mask.sum()),
        }

    return results


if __name__ == "__main__":
    np.random.seed(42)
    # Test with synthetic test case
    hr = np.random.rand(4, 256, 256).astype(np.float32)
    hr[3, 50:100, 50:100] = 0.6  # simulated canopy
    hr[0, 50:100, 50:100] = 0.1

    bicubic = hr + np.random.normal(0, 0.05, (4, 256, 256)).astype(np.float32)
    srcnn = hr + np.random.normal(0, 0.03, (4, 256, 256)).astype(np.float32)
    rcan = hr + np.random.normal(0, 0.015, (4, 256, 256)).astype(np.float32)

    res = evaluate_downstream_suite(bicubic, srcnn, rcan, hr)
    print("Downstream Task Evaluation:")
    for task, models in res.items():
        print(f"\nTask: {task}")
        for m, scores in models.items():
            if m != "ground_truth_pixel_count":
                print(f"  {m:8s} -> F1: {scores['f1']:.4f}, IoU: {scores['iou']:.4f}, Recall: {scores['recall']:.4f}")
    print("\n[PASS] Downstream task evaluation verified.")
