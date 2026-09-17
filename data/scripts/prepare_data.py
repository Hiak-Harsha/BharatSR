"""
BharatSR Data Pipeline — Phase 1
Loads opensr-test datasets, preprocesses into LR/HR patch pairs for 4x super-resolution.

CRITICAL RULES:
- Physical reflectance normalization ONLY (0-1 float). NO ImageNet mean/std.
- Bright targets (cloud, snow, specular water) may exceed 1.0 — do NOT clip.
- Train/val split by SCENE, not by patch (prevents data leakage).
- Augmentation: rotation/flip only. No color jitter (corrupts reflectance).
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_TILES_DIR = PROJECT_ROOT / "backend" / "sample_tiles"


def try_load_opensr(dataset_name: str):
    """Attempt to load an opensr-test dataset. Returns None if unavailable."""
    try:
        import opensr_test
        print(f"Loading opensr-test dataset: {dataset_name}...")
        dataset = opensr_test.load(dataset_name)
        print(f"  Loaded {dataset_name}: {type(dataset)}")
        return dataset
    except ImportError:
        print("opensr-test not installed. Will use synthetic fallback.")
        return None
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return None


def generate_synthetic_pairs(n_scenes=20, lr_size=64, scale=4, n_bands=4):
    """
    Fallback: Generate synthetic LR/HR pairs for pipeline testing.
    Uses structured patterns (not random noise) so SR has something meaningful to learn.
    This is explicitly a FALLBACK — real data from opensr-test is always preferred.
    """
    print(f"Generating {n_scenes} synthetic LR/HR scene pairs...")
    hr_size = lr_size * scale
    scenes = []

    for i in range(n_scenes):
        np.random.seed(i + 42)

        # Create a structured HR image with gradients, edges, and textures
        hr = np.zeros((n_bands, hr_size, hr_size), dtype=np.float32)

        for b in range(n_bands):
            # Base: smooth gradient
            x = np.linspace(0, 1, hr_size)
            y = np.linspace(0, 1, hr_size)
            xx, yy = np.meshgrid(x, y)
            base = (xx * 0.3 + yy * 0.3) * (0.5 + 0.5 * np.random.rand())

            # Add some "features" — circles and rectangles to simulate land cover
            cx, cy = np.random.rand(2) * hr_size
            r = 20 + np.random.rand() * 40
            mask = ((np.arange(hr_size)[:, None] - cy) ** 2 +
                    (np.arange(hr_size)[None, :] - cx) ** 2) < r ** 2
            base[mask] += 0.2 + 0.1 * np.random.rand()

            # Add edge features
            edge_pos = int(hr_size * (0.3 + 0.4 * np.random.rand()))
            base[:, edge_pos:edge_pos + 3] += 0.15

            # Add fine texture (what SR should recover)
            texture = np.random.rand(hr_size, hr_size).astype(np.float32) * 0.05
            base += texture

            hr[b] = base

        # Normalize to reflectance range [0, ~1]
        # NOTE: We allow values slightly > 1.0 for bright targets
        hr = np.clip(hr, 0, None)  # Only clip negatives
        hr = hr / (hr.max() + 1e-6) * 0.85  # Scale so most values are in [0, 0.85]

        # Simulate some bright spots that exceed 1.0 (like clouds/snow)
        if np.random.rand() > 0.7:
            bright_x, bright_y = np.random.randint(0, hr_size, 2)
            hr[:, bright_x:bright_x + 5, bright_y:bright_y + 5] = 1.05 + 0.1 * np.random.rand()

        # Generate LR by proper downsampling (area averaging, not just subsampling)
        lr = np.zeros((n_bands, lr_size, lr_size), dtype=np.float32)
        for b in range(n_bands):
            for i_lr in range(lr_size):
                for j_lr in range(lr_size):
                    i_start, j_start = i_lr * scale, j_lr * scale
                    lr[b, i_lr, j_lr] = hr[b, i_start:i_start + scale,
                                             j_start:j_start + scale].mean()

        scenes.append({"lr": lr, "hr": hr, "scene_id": i})

    return scenes


def extract_opensr_scenes(dataset, dataset_name: str):
    """
    Extract LR/HR pairs from an opensr-test dataset object.
    Handles the actual data format from the package.
    """
    scenes = []

    try:
        # opensr-test datasets are typically lists/iterables of dicts or tuples
        n_items = len(dataset) if hasattr(dataset, '__len__') else 0
        print(f"  Dataset {dataset_name} has {n_items} items")

        for idx in range(min(n_items, 100)):  # Cap at 100 scenes
            try:
                item = dataset[idx]

                # Handle different data formats from opensr-test
                if isinstance(item, dict):
                    lr = item.get("lr", item.get("LR", item.get("input", None)))
                    hr = item.get("hr", item.get("HR", item.get("target", None)))
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    lr, hr = item[0], item[1]
                else:
                    print(f"  Unexpected item format at idx {idx}: {type(item)}")
                    continue

                if lr is None or hr is None:
                    print(f"  Missing LR or HR at idx {idx}")
                    continue

                # Convert to numpy if torch tensor
                if hasattr(lr, 'numpy'):
                    lr = lr.numpy()
                if hasattr(hr, 'numpy'):
                    hr = hr.numpy()

                lr = np.array(lr, dtype=np.float32)
                hr = np.array(hr, dtype=np.float32)

                # Ensure (C, H, W) format
                if lr.ndim == 3 and lr.shape[2] <= 8:  # (H, W, C) -> (C, H, W)
                    lr = np.transpose(lr, (2, 0, 1))
                if hr.ndim == 3 and hr.shape[2] <= 8:
                    hr = np.transpose(hr, (2, 0, 1))

                # Normalize to physical reflectance [0, ~1]
                # opensr-test data may be in [0, 10000] (Sentinel-2 L2A) or [0, 1]
                if lr.max() > 10:  # Likely Sentinel-2 DN values
                    lr = lr / 10000.0
                    hr = hr / 10000.0

                # NEVER clip values > 1.0 — bright targets are physically valid
                lr = np.clip(lr, 0, None)  # Only clip negatives
                hr = np.clip(hr, 0, None)

                # Use first 4 bands (R, G, B, NIR) if more are available
                if lr.shape[0] > 4:
                    lr = lr[:4]
                if hr.shape[0] > 4:
                    hr = hr[:4]

                scenes.append({
                    "lr": lr,
                    "hr": hr,
                    "scene_id": idx,
                    "dataset": dataset_name
                })

            except Exception as e:
                print(f"  Error extracting scene {idx}: {e}")
                continue

    except Exception as e:
        print(f"Error iterating dataset {dataset_name}: {e}")

    return scenes


def tile_scene(lr, hr, lr_patch_size=64, scale=4):
    """
    Tile a scene into fixed-size LR/HR patch pairs.
    LR patches: lr_patch_size x lr_patch_size
    HR patches: (lr_patch_size * scale) x (lr_patch_size * scale)
    """
    hr_patch_size = lr_patch_size * scale
    c_lr, h_lr, w_lr = lr.shape
    c_hr, h_hr, w_hr = hr.shape

    patches = []

    # Calculate how many patches fit
    n_h = h_lr // lr_patch_size
    n_w = w_lr // lr_patch_size

    for i in range(n_h):
        for j in range(n_w):
            lr_patch = lr[:, i * lr_patch_size:(i + 1) * lr_patch_size,
                            j * lr_patch_size:(j + 1) * lr_patch_size]

            # Corresponding HR patch
            hr_i, hr_j = i * hr_patch_size, j * hr_patch_size
            if hr_i + hr_patch_size > h_hr or hr_j + hr_patch_size > w_hr:
                continue

            hr_patch = hr[:, hr_i:hr_i + hr_patch_size,
                            hr_j:hr_j + hr_patch_size]

            # Verify shapes
            if lr_patch.shape == (c_lr, lr_patch_size, lr_patch_size) and \
               hr_patch.shape == (c_hr, hr_patch_size, hr_patch_size):
                patches.append((lr_patch, hr_patch))

    return patches


def augment_patch(lr_patch, hr_patch):
    """
    Augment a patch pair with rotation and flip ONLY.
    NO color jitter — this would corrupt physical reflectance values.
    """
    augmented = [(lr_patch.copy(), hr_patch.copy())]

    # 90-degree rotations
    for k in [1, 2, 3]:
        lr_rot = np.rot90(lr_patch, k, axes=(1, 2)).copy()
        hr_rot = np.rot90(hr_patch, k, axes=(1, 2)).copy()
        augmented.append((lr_rot, hr_rot))

    # Horizontal flip
    lr_flip = np.flip(lr_patch, axis=2).copy()
    hr_flip = np.flip(hr_patch, axis=2).copy()
    augmented.append((lr_flip, hr_flip))

    # Vertical flip
    lr_flip_v = np.flip(lr_patch, axis=1).copy()
    hr_flip_v = np.flip(hr_patch, axis=1).copy()
    augmented.append((lr_flip_v, hr_flip_v))

    return augmented


def split_by_scene(scenes, val_ratio=0.2):
    """
    Split scenes into train/val by SCENE index, not by patch.
    This prevents data leakage between adjacent patches of the same scene.
    """
    n = len(scenes)
    n_val = max(1, int(n * val_ratio))

    # Deterministic shuffle
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)

    val_indices = set(indices[:n_val])
    train_scenes = [s for i, s in enumerate(scenes) if i not in val_indices]
    val_scenes = [s for i, s in enumerate(scenes) if i in val_indices]

    return train_scenes, val_scenes


def prepare_dataset(lr_patch_size=64, scale=4, augment=True, force_synthetic=False):
    """
    Main data preparation pipeline.
    Returns paths to saved train/val .npz files.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_TILES_DIR.mkdir(parents=True, exist_ok=True)

    all_scenes = []

    if not force_synthetic:
        # Try loading real data
        for dataset_name in ["spot", "naip"]:
            dataset = try_load_opensr(dataset_name)
            if dataset is not None:
                scenes = extract_opensr_scenes(dataset, dataset_name)
                all_scenes.extend(scenes)
                print(f"  Extracted {len(scenes)} scenes from {dataset_name}")

    # Fallback to synthetic if no real data available
    if len(all_scenes) == 0:
        print("\n*** Using synthetic fallback data ***")
        print("*** Install opensr-test for real satellite data ***\n")
        all_scenes = generate_synthetic_pairs(
            n_scenes=20, lr_size=lr_patch_size, scale=scale
        )
        # Synthetic scenes are already patches, no tiling needed
        is_synthetic = True
    else:
        is_synthetic = False

    print(f"\nTotal scenes: {len(all_scenes)}")

    # Split by scene
    train_scenes, val_scenes = split_by_scene(all_scenes)
    print(f"Train scenes: {len(train_scenes)}, Val scenes: {len(val_scenes)}")

    # Tile and collect patches
    train_lr, train_hr = [], []
    val_lr, val_hr = [], []

    for split_name, scenes, lr_list, hr_list in [
        ("train", train_scenes, train_lr, train_hr),
        ("val", val_scenes, val_lr, val_hr),
    ]:
        for scene in scenes:
            if is_synthetic:
                # Synthetic scenes are already patch-sized
                patches = [(scene["lr"], scene["hr"])]
            else:
                patches = tile_scene(scene["lr"], scene["hr"],
                                     lr_patch_size=lr_patch_size, scale=scale)

            for lr_p, hr_p in patches:
                if augment and split_name == "train":
                    aug_pairs = augment_patch(lr_p, hr_p)
                    for a_lr, a_hr in aug_pairs:
                        lr_list.append(a_lr)
                        hr_list.append(a_hr)
                else:
                    lr_list.append(lr_p)
                    hr_list.append(hr_p)

    if len(train_lr) == 0:
        print("ERROR: No training patches generated!")
        sys.exit(1)

    train_lr = np.array(train_lr, dtype=np.float32)
    train_hr = np.array(train_hr, dtype=np.float32)
    val_lr = np.array(val_lr, dtype=np.float32)
    val_hr = np.array(val_hr, dtype=np.float32)

    # Save
    train_path = DATA_DIR / "train.npz"
    val_path = DATA_DIR / "val.npz"

    np.savez_compressed(str(train_path), lr=train_lr, hr=train_hr)
    np.savez_compressed(str(val_path), lr=val_lr, hr=val_hr)

    # Save a few sample tiles for the demo
    n_samples = min(4, len(val_lr))
    for i in range(n_samples):
        sample_path = SAMPLE_TILES_DIR / f"sample_{i}.npz"
        np.savez_compressed(str(sample_path),
                            lr=val_lr[i], hr=val_hr[i],
                            sample_id=i)

    # Print sanity check
    print(f"\n{'='*60}")
    print(f"SANITY CHECK")
    print(f"{'='*60}")
    print(f"Train LR shape: {train_lr.shape}  (N, C, H, W)")
    print(f"Train HR shape: {train_hr.shape}")
    print(f"Val LR shape:   {val_lr.shape}")
    print(f"Val HR shape:   {val_hr.shape}")
    print(f"LR value range: [{train_lr.min():.4f}, {train_lr.max():.4f}]")
    print(f"HR value range: [{train_hr.min():.4f}, {train_hr.max():.4f}]")
    print(f"LR dtype: {train_lr.dtype}")
    print(f"HR dtype: {train_hr.dtype}")
    print(f"Scale factor: {train_hr.shape[-1] / train_lr.shape[-1]:.1f}x")
    print(f"Bands: {train_lr.shape[1]}")

    if train_lr.max() > 1.0 or train_hr.max() > 1.0:
        print(f"\nNOTE: Values > 1.0 detected — this is expected for bright")
        print(f"      targets (clouds, snow, specular water). NOT clipped.")

    print(f"\nSaved to:")
    print(f"  Train: {train_path} ({train_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Val:   {val_path} ({val_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  Samples: {SAMPLE_TILES_DIR} ({n_samples} tiles)")
    print(f"{'='*60}")

    return str(train_path), str(val_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BharatSR Data Preparation Pipeline")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic dataset generation without network downloads")
    parser.add_argument("--lr-patch-size", type=int, default=64, help="LR patch size (default: 64)")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor (default: 4)")
    parser.add_argument("--no-augment", action="store_true", help="Disable patch augmentation")
    args = parser.parse_args()

    prepare_dataset(
        lr_patch_size=args.lr_patch_size,
        scale=args.scale,
        augment=not args.no_augment,
        force_synthetic=args.synthetic
    )
