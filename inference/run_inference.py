"""
BharatSR — Production Inference CLI
Performs memory-safe tiled sliding-window inference with smooth 2D window blending.
Produces authoritative 4-band Float32 GeoTIFF with strictly preserved and scaled geospatial metadata.

Usage:
    python inference/run_inference.py --input input.tif --output output_4x.tif [--tile_size 128] [--overlap 32] [--device cpu]
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import argparse
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.models_ml.rcan import RCAN
from backend.app.models_ml.srcnn import SRCNN
from backend.app.services.inference import process_geotiff_file
from tools.validate_geotiff import validate_geotiff


def main():
    parser = argparse.ArgumentParser(description="BharatSR Production Tiled GeoTIFF Inference")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input multi-band GeoTIFF")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output super-resolved GeoTIFF")
    parser.add_argument("--model", "-m", type=str, default="rcan", choices=["rcan", "srcnn"], help="Model to use")
    parser.add_argument("--tile_size", type=int, default=128, help="Tile size in pixels (default 128)")
    parser.add_argument("--overlap", type=int, default=32, help="Tile overlap in pixels (default 32)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--verify", action="store_true", default=True, help="Automatically verify output with rasterio")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        sys.exit(1)

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"\n{'='*70}")
    print(f"BharatSR Tiled Inference Pipeline")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Model: {args.model} | Device: {device} | Tile Size: {args.tile_size} | Overlap: {args.overlap}")
    print(f"{'='*70}\n")

    # Load model
    scale_factor = 4
    n_bands = 4
    weights_dir = PROJECT_ROOT / "backend" / "weights"

    if args.model == "rcan":
        chk_path = weights_dir / "rcan_best.pth"
        if not chk_path.exists():
            print(f"Error: RCAN weights not found at {chk_path}. Run training or ablations first.")
            sys.exit(1)
        chk = torch.load(str(chk_path), map_location=device, weights_only=False)
        model = RCAN(
            n_bands=n_bands,
            n_feats=chk.get("n_feats", 36),
            n_resgroups=chk.get("n_resgroups", 3),
            n_resblocks=chk.get("n_resblocks", 3),
            scale=scale_factor,
            predict_uncertainty=True,
        ).to(device)
        model.load_state_dict(chk["model_state_dict"])
        model.eval()
    else:
        chk_path = weights_dir / "srcnn_best.pth"
        if not chk_path.exists():
            print(f"Error: SRCNN weights not found at {chk_path}.")
            sys.exit(1)
        chk = torch.load(str(chk_path), map_location=device, weights_only=False)
        model = SRCNN(n_bands=n_bands).to(device)
        model.load_state_dict(chk["model_state_dict"])
        model.eval()

    # Process GeoTIFF
    metadata = process_geotiff_file(
        input_path=str(input_path),
        output_path=str(output_path),
        model=model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        scale_factor=scale_factor,
        device=device,
    )

    print("\nInference Complete! Metadata summary:")
    print(f"  Input size: {metadata['input_size']}")
    print(f"  Output size: {metadata['output_size']}")
    print(f"  Tiles processed: {metadata['tiles_processed']}")
    print(f"  Total processing time: {metadata['total_time_s']}s ({metadata['s_per_tile']}s / tile)")
    print(f"  Preserved CRS: {metadata['crs']}")

    if args.verify:
        print("\nVerifying output GeoTIFF with rasterio...")
        is_valid = validate_geotiff(output_path)
        if not is_valid:
            print("Validation FAILED.")
            sys.exit(1)
        else:
            print("Geospatial validation PASSED with 100% compliance!")


if __name__ == "__main__":
    main()
