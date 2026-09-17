"""
BharatSR — Unified Application Launcher (FastAPI Backend + Next.js Frontend)
National Technical Research Organisation (NTRO) / SIH26142
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent

def check_environment():
    print("=" * 60)
    print(" BharatSR: Deep Learning Super-Resolution Mapping System")
    print(" NTRO Space Technology • Problem Statement SIH26142")
    print("=" * 60)

    # 1. Check Python Venv
    venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    
    if not venv_python.exists():
        print(f"[ERROR] Virtual environment python not found at {venv_python}")
        print("Please set up venv first: python -m venv venv")
        sys.exit(1)
    print(f"[OK] Python runtime: {venv_python}")

    # 2. Check Weights
    weights_dir = PROJECT_ROOT / "backend" / "weights"
    srcnn_path = weights_dir / "srcnn_best.pth"
    rcan_path = weights_dir / "rcan_best.pth"

    if srcnn_path.exists():
        print(f"[OK] SRCNN Baseline checkpoint loaded ({srcnn_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"[WARN] SRCNN checkpoint missing at {srcnn_path}")

    if rcan_path.exists():
        print(f"[OK] RCAN Dual-Head checkpoint loaded ({rcan_path.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"[WARN] RCAN checkpoint missing at {rcan_path}")

    # 3. Check Sample Tiles
    sample_dir = PROJECT_ROOT / "backend" / "sample_tiles"
    samples = list(sample_dir.glob("sample_*.npz"))
    print(f"[OK] Found {len(samples)} pre-calibrated multi-band sample tiles")

    return str(venv_python)


def main():
    venv_py = check_environment()

    print("\nStarting services...")
    # Start Backend (FastAPI)
    backend_cmd = [
        venv_py, "-m", "uvicorn", "backend.app.main:app",
        "--host", "127.0.0.1", "--port", "8000"
    ]
    print("[1/2] Launching FastAPI Backend on http://127.0.0.1:8000...")
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(PROJECT_ROOT),
        shell=False,
    )

    # Start Frontend (Next.js)
    frontend_dir = PROJECT_ROOT / "frontend"
    print("[2/2] Launching Next.js Frontend on http://localhost:3000...")
    # Use npm run start or npm run dev
    frontend_cmd = "npm run start -- -p 3000" if (frontend_dir / ".next").exists() else "npm run dev"
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=str(frontend_dir),
        shell=True,
    )

    print("\n" + "-" * 60)
    print("BharatSR is live!")
    print("  • Web Interface: http://localhost:3000")
    print("  • Backend API:   http://127.0.0.1:8000")
    print("  • API Docs:      http://127.0.0.1:8000/docs")
    print("-" * 60)
    print("Press Ctrl+C to terminate both servers.\n")

    # Give services a few seconds to warm up, then open browser
    time.sleep(3)
    try:
        webbrowser.open("http://localhost:3000")
    except Exception:
        pass

    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("Backend stopped unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("Frontend stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nShutting down BharatSR services gracefully...")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        try:
            backend_proc.wait(timeout=3)
            frontend_proc.wait(timeout=3)
        except Exception:
            backend_proc.kill()
            frontend_proc.kill()
        print("Done.")

if __name__ == "__main__":
    main()
