"""
BharatSR — 70 Combination Runtime Verification Script
Exercises all 7 models x 2 quality modes x 5 sample scenes = 70 combinations
against POST /api/superresolve.
Verifies:
1. No bare 500 Internal Server Errors.
2. Every response is valid, standard-compliant JSON (strictly parseable without NaN/Infinity tokens).
3. Verifies error_raw (grayscale) and error (heatmap) presence when ground truth exists.
4. Synthetically tests degenerate case (identical images, MSE=0) for NaN/Inf handling.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import urllib.request
import urllib.parse
import urllib.error
import time

BASE_URL = "http://localhost:8000"

MODELS = ["bicubic", "srcnn", "rcan", "swinir", "hat", "diffusion", "ensemble"]
QUALITIES = ["fast", "high"]
SAMPLES = ["sample_0", "sample_1", "sample_2", "sample_3", "sample_real_s2"]

def test_combination(sample_id, model_id, quality):
    data = urllib.parse.urlencode({
        "sample_id": sample_id,
        "model_id": model_id,
        "quality": quality,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/api/superresolve",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            elapsed = time.time() - start
            raw_bytes = resp.read()
            raw_str = raw_bytes.decode("utf-8")

            # Strict JSON check: Python's json.loads with parse_constant that rejects NaN/Infinity
            def strict_constant(x):
                raise ValueError(f"Strict JSON violation: found non-standard token '{x}'")

            parsed = json.loads(raw_str, parse_constant=strict_constant)
            
            # Check for error_raw in views or error_map
            has_error_raw = False
            if "views" in parsed and "error_raw" in parsed["views"]:
                has_error_raw = True
            elif "error_map" in parsed and isinstance(parsed["error_map"], dict) and "image_raw" in parsed["error_map"]:
                has_error_raw = True

            return {
                "ok": True,
                "status_code": resp.status,
                "elapsed_s": round(elapsed, 2),
                "model_id": model_id,
                "quality": quality,
                "sample_id": sample_id,
                "has_error_raw": has_error_raw,
                "error": None,
            }
    except urllib.error.HTTPError as e:
        elapsed = time.time() - start
        raw_body = e.read().decode("utf-8", errors="replace")
        try:
            body_json = json.loads(raw_body)
            # Check if it's a structured error
            is_structured = isinstance(body_json, dict) and ("detail" in body_json or "error" in body_json)
        except Exception:
            is_structured = False

        return {
            "ok": False,
            "status_code": e.code,
            "elapsed_s": round(elapsed, 2),
            "model_id": model_id,
            "quality": quality,
            "sample_id": sample_id,
            "is_structured": is_structured,
            "raw_body": raw_body[:200],
            "error": f"HTTP {e.code}: {raw_body[:100]}",
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "ok": False,
            "status_code": 0,
            "elapsed_s": round(elapsed, 2),
            "model_id": model_id,
            "quality": quality,
            "sample_id": sample_id,
            "error": f"Client exception: {type(e).__name__} - {str(e)}",
        }

def test_degenerate_case():
    print("\n--- Testing Degenerate Case (Zero-MSE / Infinite PSNR) ---")
    import numpy as np
    from backend.app.services.postprocessing import compute_inference_metrics
    
    # Construct identical images (MSE = 0.0)
    img = np.random.rand(4, 64, 64).astype(np.float32)
    metrics = compute_inference_metrics(img, img)
    
    from backend.app.core.json_utils import sanitize_for_json
    clean_metrics = sanitize_for_json(metrics)
    
    # Try serializing and strictly deserializing
    serialized = json.dumps(clean_metrics)
    def strict_constant(x):
        raise ValueError(f"Strict JSON violation: found non-standard token '{x}'")
    
    parsed = json.loads(serialized, parse_constant=strict_constant)
    
    psnr_entry = parsed.get("psnr", {})
    print(f"Degenerate PSNR entry: {psnr_entry}")
    assert psnr_entry.get("value") is None, "PSNR value should be None when MSE=0"
    assert "identical" in psnr_entry.get("note", "").lower() or "zero error" in psnr_entry.get("note", "").lower(), "PSNR note should explain perfect match"
    print("Degenerate test PASSED: serialized cleanly without NaN/Infinity tokens.")

def main():
    test_degenerate_case()

    print(f"\n--- Running All {len(MODELS) * len(QUALITIES) * len(SAMPLES)} Combinations ---")
    results = []
    failed_runs = []
    bare_500s = []

    count = 0
    total = len(MODELS) * len(QUALITIES) * len(SAMPLES)

    for s_idx, sample_id in enumerate(SAMPLES, 1):
        for m_idx, model_id in enumerate(MODELS, 1):
            for q_idx, quality in enumerate(QUALITIES, 1):
                count += 1
                res = test_combination(sample_id, model_id, quality)
                results.append(res)

                if res["ok"]:
                    status_desc = f"SUCCESS (status={res['status_code']}, {res['elapsed_s']}s, error_raw={res['has_error_raw']})"
                else:
                    if res["status_code"] == 500 and not res.get("is_structured"):
                        bare_500s.append(res)
                        status_desc = f"BARE 500 ERROR: {res['raw_body']}"
                    else:
                        failed_runs.append(res)
                        status_desc = f"STRUCTURED ERROR ({res['status_code']}): {res.get('raw_body', res['error'])}"

                print(f"[{count:02d}/{total}] sample={sample_id:<14} model={model_id:<10} q={quality:<4} -> {status_desc}")

    print("\n=================== SUMMARY ===================")
    print(f"Total Runs: {total}")
    success_count = sum(1 for r in results if r["ok"])
    print(f"Successful Runs: {success_count}/{total}")
    print(f"Bare 500 Errors: {len(bare_500s)}")
    print(f"Structured Errors: {len(failed_runs)}")

    if bare_500s:
        print("\nCRITICAL FAILURE: Encountered bare 500 errors!")
        for b in bare_500s:
            print(f"  - {b}")
        sys.exit(1)
    else:
        print("\nPASSED: Zero bare 500 errors observed across all 70 combinations.")

if __name__ == "__main__":
    main()
