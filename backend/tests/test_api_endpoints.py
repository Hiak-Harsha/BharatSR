"""
BharatSR — Backend API Integration Tests
Tests /api/health, /api/models, /api/samples, and /api/superresolve.
"""

import urllib.request
import urllib.parse
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_all():
    print(f"Testing BharatSR API at {BASE_URL}...\n")

    # 1. Health
    req = urllib.request.Request(f"{BASE_URL}/api/health")
    with urllib.request.urlopen(req) as resp:
        health = json.loads(resp.read().decode())
        print(f"[PASS] Health check: {health}")
        assert health["status"] == "ok"

    # 2. Models
    req = urllib.request.Request(f"{BASE_URL}/api/models")
    with urllib.request.urlopen(req) as resp:
        models = json.loads(resp.read().decode())
        print(f"[PASS] Models list: {[m['id'] for m in models['models']]}")
        assert len(models["models"]) > 0

    # 3. Samples
    req = urllib.request.Request(f"{BASE_URL}/api/samples")
    with urllib.request.urlopen(req) as resp:
        samples = json.loads(resp.read().decode())
        print(f"[PASS] Samples count: {len(samples['samples'])}")
        assert len(samples["samples"]) > 0
        sample_id = samples["samples"][0]["id"]
        print(f"       Using first sample: {sample_id}")

    # 4. Superresolve with sample_id
    data = urllib.parse.urlencode({
        "sample_id": sample_id,
        "model_id": "srcnn",
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/api/superresolve",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode())
        print(f"[PASS] Superresolve success!")
        print(f"       Inference time: {result['inference_time_s']}s")
        print(f"       Input shape:    {result['input']['shape']}")
        print(f"       Output shape:   {result['output']['shape']}")
        print(f"       Metrics:")
        for k, v in result["metrics"].items():
            if isinstance(v, dict):
                print(f"         {k}: {v.get('value')} ({v.get('rating')}) - {v.get('description')}")
            else:
                print(f"         {k}: {v}")

    # 5. Superresolve with RCAN + Uncertainty head
    data_rcan = urllib.parse.urlencode({
        "sample_id": sample_id,
        "model_id": "rcan",
    }).encode("utf-8")

    req_rcan = urllib.request.Request(
        f"{BASE_URL}/api/superresolve",
        data=data_rcan,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    with urllib.request.urlopen(req_rcan) as resp:
        result_rcan = json.loads(resp.read().decode())
        print(f"[PASS] RCAN Superresolve success!")
        print(f"       Inference time: {result_rcan['inference_time_s']}s")
        print(f"       Input shape:    {result_rcan['input']['shape']}")
        print(f"       Output shape:   {result_rcan['output']['shape']}")
        assert "uncertainty" in result_rcan, "Expected uncertainty in RCAN response"
        u_summary = result_rcan["uncertainty"]["summary"]
        print(f"       Uncertainty Map: mean_sigma={u_summary['mean_sigma']:.4f}, max_sigma={u_summary['max_sigma']:.4f}")

    print("\nALL API ENDPOINTS TESTED SUCCESSFULLY! [OK]")

if __name__ == "__main__":
    test_all()
