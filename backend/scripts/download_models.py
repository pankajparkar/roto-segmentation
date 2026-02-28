#!/usr/bin/env python3
"""
Download SAM2 model checkpoints.

Usage:
    python scripts/download_models.py [model_size]

Model sizes:
    tiny   - Fastest, lowest quality (~39M params, ~150MB)
    small  - Good balance for 16GB RAM (~46M params, ~180MB) [DEFAULT]
    base+  - Better quality (~81M params, ~320MB)
    large  - Best quality, needs 32GB+ RAM (~224M params, ~900MB)
"""

import sys
import os
from pathlib import Path
import urllib.request
import hashlib

# Model URLs from Meta's SAM2 release
MODELS = {
    "tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "filename": "sam2_hiera_tiny.pt",
        "size_mb": 150,
    },
    "small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "filename": "sam2_hiera_small.pt",
        "size_mb": 180,
    },
    "base+": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "filename": "sam2_hiera_base_plus.pt",
        "size_mb": 320,
    },
    "large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "filename": "sam2_hiera_large.pt",
        "size_mb": 900,
    },
}


def download_with_progress(url: str, dest: Path, expected_size_mb: int):
    """Download file with progress indicator."""
    print(f"Downloading from: {url}")
    print(f"Expected size: ~{expected_size_mb} MB")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 // total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            bar = "=" * (percent // 2) + ">" + " " * (50 - percent // 2)
            print(f"\r[{bar}] {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
        print()  # New line after progress bar
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False


def main():
    # Determine which model to download
    model_size = "small"  # Default for 16GB M1 Mac

    if len(sys.argv) > 1:
        model_size = sys.argv[1].lower()
        if model_size == "base":
            model_size = "base+"

    if model_size not in MODELS:
        print(f"Unknown model size: {model_size}")
        print(f"Available: {', '.join(MODELS.keys())}")
        return 1

    model_info = MODELS[model_size]

    # Create models directory
    script_dir = Path(__file__).parent
    models_dir = script_dir.parent / "models"
    models_dir.mkdir(exist_ok=True)

    dest_path = models_dir / model_info["filename"]

    # Check if already downloaded
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"Model already exists: {dest_path}")
        print(f"Size: {size_mb:.1f} MB")

        response = input("Re-download? [y/N] ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return 0

    print(f"\n{'='*60}")
    print(f"Downloading SAM2 model: {model_size}")
    print(f"{'='*60}\n")

    success = download_with_progress(
        model_info["url"],
        dest_path,
        model_info["size_mb"],
    )

    if success:
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"\nDownload complete!")
        print(f"Saved to: {dest_path}")
        print(f"Size: {size_mb:.1f} MB")

        # Update .env if needed
        env_file = script_dir.parent / ".env"
        if env_file.exists():
            print(f"\nUpdate your .env file:")
            print(f"  SAM2_MODEL={model_info['filename']}")
        else:
            print(f"\nCreate .env with:")
            print(f"  SAM2_MODEL={model_info['filename']}")
            print(f"  DEVICE=mps")

        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
