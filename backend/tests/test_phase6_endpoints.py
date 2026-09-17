"""
Test script for Phase 6 API endpoints:
- Multi-spectral views
- Multi-model comparison (/api/compare)
- Async job processing (/api/superresolve/async and /api/jobs/{job_id})
- GeoTIFF and JSON Report exports
"""

import sys
import time
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print(f"Testing BharatSR Phase 6 API at {BASE_URL}...\n")

    # 1. Health
    r = requests.get(f"{BASE_URL}/api/health", timeout=5)
    assert r.status_code == 200, f"Health failed: {r.text}"
    print(f"[PASS] Health check: {r.json()}")

    # 2. Samples and multi-spectral views
    r = requests.get(f"{BASE_URL}/api/samples", timeout=5)
    assert r.status_code == 200
    samples = r.json()["samples"]
    assert len(samples) > 0
    sample_id = samples[0]["id"]
    views = samples[0].get("views", {})
    assert "rgb" in views and "cir" in views and "ndvi" in views
    print(f"[PASS] Samples check: {len(samples)} found. Sample '{sample_id}' has views: {list(views.keys())}")

    # 3. Superresolve with multi-spectral views
    payload = {"sample_id": sample_id, "model_id": "rcan"}
    r = requests.post(f"{BASE_URL}/api/superresolve", data=payload, timeout=15)
    assert r.status_code == 200, f"Superresolve failed: {r.text}"
    data = r.json()
    assert "views" in data["output"]
    assert "ndvi" in data["output"]["views"]
    assert "cir" in data["output"]["views"]
    assert "uncertainty" in data
    print(f"[PASS] Single SR (RCAN): latency={data['inference_time_s']}s, output_views={list(data['output']['views'].keys())}")

    # 4. Multi-Model Comparison (/api/compare)
    print("Testing /api/compare...")
    t0 = time.time()
    r = requests.post(f"{BASE_URL}/api/compare", data={"sample_id": sample_id}, timeout=20)
    assert r.status_code == 200, f"Compare failed: {r.text}"
    comp_data = r.json()
    assert "models" in comp_data
    assert "bicubic" in comp_data["models"]
    assert "srcnn" in comp_data["models"]
    assert "rcan" in comp_data["models"]
    assert "comparison_table" in comp_data
    print(f"[PASS] Multi-model compare ({time.time() - t0:.2f}s):")
    for row in comp_data["comparison_table"]:
        print(f"       {row['metric']}: Bicubic={row.get('bicubic')}, SRCNN={row.get('srcnn')}, RCAN={row.get('rcan')} -> Best: {row.get('best_model')}")

    # 5. Async Job Submission & Polling
    print("\nTesting Async Job Workflow...")
    r = requests.post(f"{BASE_URL}/api/superresolve/async", data={"sample_id": sample_id, "model_id": "rcan"}, timeout=5)
    assert r.status_code == 200
    job_info = r.json()
    job_id = job_info["job_id"]
    print(f"[PASS] Submitted async job: {job_id}")

    # Poll status
    completed = False
    for _ in range(20):
        time.sleep(0.3)
        r = requests.get(f"{BASE_URL}/api/jobs/{job_id}", timeout=5)
        assert r.status_code == 200
        status_data = r.json()
        if status_data["status"] == "completed":
            completed = True
            print(f"[PASS] Async job {job_id} completed successfully in {status_data.get('inference_time_s')}s!")
            assert "result" in status_data
            break
        elif status_data["status"] == "failed":
            raise RuntimeError(f"Job failed: {status_data.get('error_message')}")

    assert completed, "Job did not complete in time"

    # 6. List jobs
    r = requests.get(f"{BASE_URL}/api/jobs", timeout=5)
    assert r.status_code == 200
    jobs = r.json().get("jobs", [])
    assert len(jobs) > 0
    print(f"[PASS] Jobs list: {len(jobs)} jobs tracked in SQLite")

    # 7. GeoTIFF Export
    print("\nTesting GeoTIFF Export...")
    r = requests.get(f"{BASE_URL}/api/export/geotiff?sample_id={sample_id}&model_id=rcan", timeout=15)
    assert r.status_code == 200
    assert r.headers.get("content-type") == "image/tiff"
    tiff_bytes = r.content
    assert len(tiff_bytes) > 50000, f"TIFF too small: {len(tiff_bytes)} bytes"
    # Verify it starts with TIFF magic bytes (II\x2a\x00 or MM\x00\x2a)
    assert tiff_bytes[:2] in (b"II", b"MM"), "Invalid TIFF header"
    print(f"[PASS] GeoTIFF export verified: {len(tiff_bytes):,} bytes, valid header ({tiff_bytes[:2].decode('ascii')})")

    # 8. Report Export
    r = requests.get(f"{BASE_URL}/api/export/report?sample_id={sample_id}&model_id=rcan", timeout=15)
    assert r.status_code == 200
    report = r.json()
    assert report["problem_statement"] == "SIH26142 - Deep Learning Super-Resolution Mapping for Medium-Resolution Satellite Imagery"
    assert "metrics" in report
    assert "spectral_integrity_compliance" in report
    print(f"[PASS] Evaluation Report export verified: Title='{report['title']}'")

    print("\nALL PHASE 6 API ENDPOINTS PASSED WITH 100% SUCCESS! [OK]\n")

if __name__ == "__main__":
    run_tests()
