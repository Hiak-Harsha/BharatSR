import os
import sys
import time
import requests
from pathlib import Path

def download_dataset(dataset_name="spain_urban", version_key="021"):
    dest_dir = Path(os.path.expanduser("~")) / ".config" / "opensr_test"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / f"{dataset_name}.pkl"
    temp_file = dest_dir / f"{dataset_name}.pkl.tmp"

    url = f"https://huggingface.co/datasets/isp-uv-es/opensr-test/resolve/main/{version_key}/{dataset_name}/{dataset_name}.pkl"
    print(f"Downloading {dataset_name} from {url}...")
    
    t0 = time.time()
    resp = requests.get(url, stream=True, timeout=30)
    total_len = int(resp.headers.get("content-length", 0))
    print(f"Total size: {total_len / (1024*1024):.2f} MB")

    downloaded = 0
    with open(temp_file, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (10 * 1024 * 1024) < 1024 * 1024:
                    elapsed = time.time() - t0
                    speed = (downloaded / (1024*1024)) / max(elapsed, 0.1)
                    print(f"  {downloaded / (1024*1024):.1f} / {total_len / (1024*1024):.1f} MB ({speed:.2f} MB/s)")

    if temp_file.exists():
        if dest_file.exists():
            dest_file.unlink()
        temp_file.rename(dest_file)
        print(f"Saved {dataset_name}.pkl ({dest_file.stat().st_size / (1024*1024):.2f} MB) in {time.time()-t0:.1f}s")
        return dest_file

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "spain_urban"
    download_dataset(name)
