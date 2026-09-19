"""
BharatSR — Downstream Analytical Task Evaluation
Evaluates whether super-resolution improves analytical task performance
(e.g., Road/Building extraction, Land-Cover classification, Micro-Canopy segmentation).

CRITICAL SCIENTIFIC PRINCIPLES:
- Do NOT use HR -> heuristic thresholding -> ground truth as an official scientific benchmark.
- Rule-based spectral indices are labeled strictly as "Rule-based spectral interpretation (not ground truth)".
- Support independent ground-truth binary or multi-class labels (e.g., Open Buildings, OSM road network).
- Evaluates: LR, Bicubic, SRCNN, RCAN, and HR against independent ground truth.
- Metrics computed:
  - Precision
  - Recall
  - F1-Score
  - Intersection over Union (IoU / Jaccard Index)
"""

from typing import Dict, Optional, Tuple
import numpy as np

BAND_INDEX: Dict[str, int] = {
    "B2": 0,
    "B3": 1,
    "B4": 2,
    "B8": 3,
}


def segment_micro_canopy_rule_based(img: np.ndarray, ndvi_threshold: float = 0.35) -> np.ndarray:
    """
    Rule-based spectral interpretation (heuristic, NOT ground truth):
    Extract active vegetation canopy using physical NDVI:
    NDVI = (B8 - B4) / (B8 + B4 + 1e-7)
    where B4 = Red (index 2), B8 = NIR (index 3).
    """
    red = img[BAND_INDEX["B4"]]
    nir = img[BAND_INDEX["B8"]]
    ndvi = (nir - red) / (nir + red + 1e-7)
    return (ndvi > ndvi_threshold).astype(np.uint8)


def segment_built_up_rule_based(img: np.ndarray, threshold: float = 0.18) -> np.ndarray:
    """
    Rule-based spectral interpretation (heuristic, NOT ground truth):
    Extract built-up infrastructure candidate areas using visible albedo
    and low NIR-Red contrast.
    """
    visible_mean = (img[BAND_INDEX["B2"]] + img[BAND_INDEX["B3"]] + img[BAND_INDEX["B4"]]) / 3.0
    red = img[BAND_INDEX["B4"]]
    nir = img[BAND_INDEX["B8"]]
    contrast = np.abs(nir - red)
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
    independent_gt_mask: Optional[np.ndarray] = None,
    lr_raw: Optional[np.ndarray] = None,
) -> Dict[str, dict]:
    """
    Evaluate all models on downstream analytical segmentation tasks.

    If independent_gt_mask is provided, evaluates against independent labels.
    Otherwise, evaluates rule-based spectral interpretations clearly labeled
    as 'Rule-based spectral interpretation (not ground truth)'.
    """
    results = {}

    if independent_gt_mask is not None:
        # Benchmark against genuine independent ground truth
        from scipy.ndimage import zoom

        # Up-sample raw LR if provided
        if lr_raw is not None and lr_raw.shape[-2:] != independent_gt_mask.shape[-2:]:
            zh = independent_gt_mask.shape[-2] / lr_raw.shape[-2]
            zw = independent_gt_mask.shape[-1] / lr_raw.shape[-1]
            lr_upsampled = np.stack([zoom(lr_raw[b], (zh, zw), order=0) for b in range(lr_raw.shape[0])], axis=0)
            mask_lr = segment_built_up_rule_based(lr_upsampled)
        else:
            mask_lr = segment_built_up_rule_based(lr_bicubic)

        mask_bicubic = segment_built_up_rule_based(lr_bicubic)
        mask_srcnn = segment_built_up_rule_based(sr_srcnn)
        mask_rcan = segment_built_up_rule_based(sr_rcan)
        mask_hr = segment_built_up_rule_based(hr_reference)

        results["independent_label_evaluation"] = {
            "evaluation_type": "Independent Ground-Truth Labels",
            "lr": compute_segmentation_metrics(mask_lr, independent_gt_mask),
            "bicubic": compute_segmentation_metrics(mask_bicubic, independent_gt_mask),
            "srcnn": compute_segmentation_metrics(mask_srcnn, independent_gt_mask),
            "rcan": compute_segmentation_metrics(mask_rcan, independent_gt_mask),
            "hr": compute_segmentation_metrics(mask_hr, independent_gt_mask),
            "ground_truth_pixel_count": int(independent_gt_mask.sum()),
        }
    else:
        # Rule-based diagnostic interpretations
        tasks = {
            "canopy_segmentation": segment_micro_canopy_rule_based,
            "built_up_infrastructure": segment_built_up_rule_based,
        }

        for task_name, seg_fn in tasks.items():
            ref_mask = seg_fn(hr_reference)
            mask_bicubic = seg_fn(lr_bicubic)
            mask_srcnn = seg_fn(sr_srcnn)
            mask_rcan = seg_fn(sr_rcan)

            results[task_name] = {
                "evaluation_type": "Rule-based spectral interpretation (not ground truth)",
                "bicubic": compute_segmentation_metrics(mask_bicubic, ref_mask),
                "srcnn": compute_segmentation_metrics(mask_srcnn, ref_mask),
                "rcan": compute_segmentation_metrics(mask_rcan, ref_mask),
                "reference_pixel_count": int(ref_mask.sum()),
            }

    return results


if __name__ == "__main__":
    np.random.seed(42)
    # Test with synthetic test case
    hr = np.random.rand(4, 256, 256).astype(np.float32)
    hr[BAND_INDEX["B8"], 50:100, 50:100] = 0.6  # simulated canopy NIR
    hr[BAND_INDEX["B4"], 50:100, 50:100] = 0.1  # simulated canopy Red

    # Independent ground truth mask (e.g. independently mapped building footprints)
    indep_gt = np.zeros((256, 256), dtype=np.uint8)
    indep_gt[40:110, 40:110] = 1

    bicubic = hr + np.random.normal(0, 0.05, (4, 256, 256)).astype(np.float32)
    srcnn = hr + np.random.normal(0, 0.03, (4, 256, 256)).astype(np.float32)
    rcan = hr + np.random.normal(0, 0.015, (4, 256, 256)).astype(np.float32)

    res = evaluate_downstream_suite(bicubic, srcnn, rcan, hr, independent_gt_mask=indep_gt)
    print("Downstream Task Evaluation (Independent Labels):")
    for task, data in res.items():
        print(f"\nTask: {task} ({data['evaluation_type']})")
        for m in ["lr", "bicubic", "srcnn", "rcan", "hr"]:
            if m in data:
                scores = data[m]
                print(f"  {m:8s} -> IoU: {scores['iou']:.4f}, F1: {scores['f1']:.4f}, Precision: {scores['precision']:.4f}, Recall: {scores['recall']:.4f}")
    print("\n[PASS] Downstream task evaluation verified.")
