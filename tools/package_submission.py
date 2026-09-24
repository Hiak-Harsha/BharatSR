#!/usr/bin/env python3
"""
BharatSR Submission Packaging Tool (SIH26142)

Creates a deterministic, clean, bloat-free ZIP archive of the repository
for Smart India Hackathon submission.

Excludes:
- venv / virtual environments
- frontend/node_modules
- frontend/.next
- .git repository
- .pytest_cache, __pycache__, *.pyc
- Existing submission ZIPs
- Scratch/temporary development artifacts
"""

import os
import sys
import zipfile
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP_NAME = "BharatSR_SIH26142_Submission(4).zip"
OUTPUT_ZIP_PATH = PROJECT_ROOT / OUTPUT_ZIP_NAME

EXCLUDE_DIRS = {
    ".git",
    "venv",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    ".next",
    ".system_generated",
    "scratch",
    ".idea",
    ".vscode",
    "internal-notes",
    "runs",       # Exclude transient inference/evaluation run logs and caches
    "processed",  # Exclude large raw training npz datasets (train.npz is 774MB)
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".tmp",
}

EXCLUDE_FILES = {
    ".DS_Store",
    "Thumbs.db",
    "bharatsr.db-journal",
}


def should_exclude(rel_path_str: str, profile: str = "standard") -> bool:
    path_obj = Path(rel_path_str)
    parts = path_obj.parts

    # Standard directory exclusions
    for p in parts:
        if p in EXCLUDE_DIRS:
            return True
        if p.endswith(".egg-info"):
            return True

    filename = path_obj.name
    if filename in EXCLUDE_FILES:
        return True

    suffix = path_obj.suffix.lower()
    if suffix in EXCLUDE_EXTENSIONS:
        return True

    # Do not include existing or prior submission ZIPs
    if filename.startswith("BharatSR_SIH26142_Submission") and suffix == ".zip":
        return True

    # Profile-specific filtering
    posix_rel = path_obj.as_posix()
    if profile == "email":
        # Keep zip under 20MB for email attachments (Gmail limit 25MB)
        # Keep primary model rcan_best.pth, omit ablation checkpoints (saving ~7MB)
        if "rcan_ablation_" in filename:
            return True
        # Exclude bulky raw comparison PNGs in data/visualizations (saving ~17MB)
        if posix_rel.startswith("data/visualizations/") and suffix == ".png":
            return True

    elif profile == "minimal":
        # Ultra-lightweight: code, configs, and docs only
        if suffix in {".pth", ".npz", ".tif", ".tiff"}:
            return True
        if posix_rel.startswith("data/visualizations/"):
            return True

    elif profile == "full":
        # Full profile includes test.npz (27.6MB) for immediate local offline testing
        # Still excludes 774MB train.npz and runs/
        if filename in {"train.npz", "val.npz"}:
            return True

    return False


def package_repository(profile: str = "standard", custom_output_name: str = None):
    profile_names = {
        "standard": "BharatSR_SIH26142_Submission.zip",
        "email": "BharatSR_SIH26142_Submission_EmailSafe.zip",
        "full": "BharatSR_SIH26142_Submission_Full.zip",
        "minimal": "BharatSR_SIH26142_Submission_Minimal.zip",
    }
    output_filename = custom_output_name or profile_names.get(profile, f"BharatSR_SIH26142_Submission_{profile}.zip")
    output_path = PROJECT_ROOT / output_filename

    print("==================================================")
    print("  BharatSR (SIH26142) Submission Packaging Tool   ")
    print("==================================================")
    print(f"Profile:        {profile.upper()}")
    print(f"Source Root:    {PROJECT_ROOT}")
    print(f"Target Archive: {output_path}\n")

    files_to_add = []
    total_uncompressed_bytes = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Filter directories in-place to avoid recursing into excluded ones
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")]

        for f in files:
            full_path = Path(root) / f
            rel_path = full_path.relative_to(PROJECT_ROOT)
            rel_path_str = str(rel_path).replace("\\", "/")

            if not should_exclude(rel_path_str, profile=profile):
                sz = full_path.stat().st_size
                files_to_add.append((full_path, rel_path_str, sz))
                total_uncompressed_bytes += sz

    print(f"Discovered {len(files_to_add):,} clean project files to package.")
    print(f"Total uncompressed size: {total_uncompressed_bytes / (1024 * 1024):.2f} MB")

    if output_path.exists():
        os.remove(output_path)

    print("\nArchiving files with ZIP_DEFLATED compression (level 9)...")
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for full_p, rel_p, _ in files_to_add:
            zf.write(full_p, arcname=rel_p)

    archive_size_bytes = output_path.stat().st_size
    archive_size_mb = archive_size_bytes / (1024 * 1024)

    # Compute SHA-256
    hasher = hashlib.sha256()
    with open(output_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    zip_sha256 = hasher.hexdigest()

    # Integrity test
    print("Verifying archive integrity via zipfile.testzip()...")
    with zipfile.ZipFile(output_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            print(f"ERROR: Corrupted file detected in archive: {bad_file}")
            sys.exit(1)
        archive_file_count = len(zf.infolist())

    print("\n==================================================")
    print("             SUBMISSION ARCHIVE READY             ")
    print("==================================================")
    print(f"Archive Name:     {output_filename}")
    print(f"Archive Location: {output_path}")
    print(f"Files Packaged:   {archive_file_count:,} files")
    print(f"Compressed Size:  {archive_size_mb:.2f} MB ({archive_size_bytes:,} bytes)")
    print(f"Compression:      {(1 - archive_size_bytes / total_uncompressed_bytes) * 100:.1f}% space saved")
    print(f"SHA-256 Checksum: {zip_sha256}")
    print("==================================================\n")

    return output_path, zip_sha256


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Package BharatSR for SIH26142 submission.")
    parser.add_argument(
        "--profile",
        choices=["standard", "email", "full", "minimal"],
        default="standard",
        help="Packaging profile: 'standard' (~32MB, portal upload), 'email' (<20MB, email attachment), 'full' (~60MB, with test set), or 'minimal' (<1MB, code only)."
    )
    parser.add_argument("--out", type=str, default=None, help="Custom output archive filename.")
    args = parser.parse_args()

    package_repository(profile=args.profile, custom_output_name=args.out)
