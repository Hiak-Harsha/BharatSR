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


def should_exclude(rel_path_str: str) -> bool:
    parts = Path(rel_path_str).parts
    for p in parts:
        if p in EXCLUDE_DIRS:
            return True
        if p.endswith(".egg-info"):
            return True

    filename = Path(rel_path_str).name
    if filename in EXCLUDE_FILES:
        return True

    suffix = Path(rel_path_str).suffix.lower()
    if suffix in EXCLUDE_EXTENSIONS:
        return True

    # Do not include existing or prior submission ZIPs
    if filename.startswith("BharatSR_SIH26142_Submission") and suffix == ".zip":
        return True

    return False


def package_repository():
    print("==================================================")
    print("  BharatSR (SIH26142) Submission Packaging Tool   ")
    print("==================================================")
    print(f"Source Root: {PROJECT_ROOT}")
    print(f"Target Archive: {OUTPUT_ZIP_PATH}\n")

    files_to_add = []
    total_uncompressed_bytes = 0

    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Filter directories in-place to avoid recursing into excluded ones
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")]

        for f in files:
            full_path = Path(root) / f
            rel_path = full_path.relative_to(PROJECT_ROOT)
            rel_path_str = str(rel_path).replace("\\", "/")

            if not should_exclude(rel_path_str):
                sz = full_path.stat().st_size
                files_to_add.append((full_path, rel_path_str, sz))
                total_uncompressed_bytes += sz

    print(f"Discovered {len(files_to_add):,} clean project files to package.")
    print(f"Total uncompressed size: {total_uncompressed_bytes / (1024 * 1024):.2f} MB")

    if OUTPUT_ZIP_PATH.exists():
        os.remove(OUTPUT_ZIP_PATH)

    print("\nArchiving files with ZIP_DEFLATED compression...")
    with zipfile.ZipFile(OUTPUT_ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for full_p, rel_p, _ in files_to_add:
            zf.write(full_p, arcname=rel_p)

    archive_size_bytes = OUTPUT_ZIP_PATH.stat().st_size
    archive_size_mb = archive_size_bytes / (1024 * 1024)

    # Compute SHA-256
    hasher = hashlib.sha256()
    with open(OUTPUT_ZIP_PATH, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    zip_sha256 = hasher.hexdigest()

    # Integrity test
    print("\nVerifying archive integrity via zipfile.testzip()...")
    with zipfile.ZipFile(OUTPUT_ZIP_PATH, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            print(f"ERROR: Corrupted file detected in archive: {bad_file}")
            sys.exit(1)
        archive_file_count = len(zf.infolist())

    print("\n==================================================")
    print("             SUBMISSION ARCHIVE READY             ")
    print("==================================================")
    print(f"Archive Name:     {OUTPUT_ZIP_NAME}")
    print(f"Archive Location: {OUTPUT_ZIP_PATH}")
    print(f"Files Packaged:   {archive_file_count:,} files")
    print(f"Compressed Size:  {archive_size_mb:.2f} MB ({archive_size_bytes:,} bytes)")
    print(f"Compression:      {(1 - archive_size_bytes / total_uncompressed_bytes) * 100:.1f}% space saved")
    print(f"SHA-256 Checksum: {zip_sha256}")
    print("==================================================\n")

    return OUTPUT_ZIP_PATH, zip_sha256


if __name__ == "__main__":
    package_repository()
