"""
AUTSL (Turkish Sign Language) Dataset Downloader

Note: AUTSL usually requires registration at:
https://chalearnlap.cvc.uab.cat/dataset/40/description/

This script handles metadata preparation and provides a CLI for automated
download if a direct mirror or temporary access link is provided.
"""

import os
import sys
import argparse
import urllib.request
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

RAW_DATA_DIR = os.path.join("datasets", "raw_videos", "AUTSL")
METADATA_DIR = os.path.join("datasets", "metadata", "AUTSL")

# Publicly discoverable metadata (labels)
METADATA_URL = "https://raw.githubusercontent.com/neccmettin/AUTSL/master/Labels/AUTSL_labels.csv"

def download_metadata():
    """Download the AUTSL labels/metadata."""
    os.makedirs(METADATA_DIR, exist_ok=True)
    out_path = os.path.join(METADATA_DIR, "Labels.csv")
    
    print(f"  Downloading AUTSL metadata from: {METADATA_URL}")
    try:
        urllib.request.urlretrieve(METADATA_URL, out_path)
        print(f"  Metadata saved to: {out_path}")
        return True
    except Exception as e:
        print(f"  Failed to download metadata: {e}")
        return False

def progress_hook(count, block_size, total_size):
    """Show download progress bar."""
    percent = min(int(count * block_size * 100 / max(total_size, 1)), 100)
    bar = "█" * (percent // 2) + "░" * (50 - percent // 2)
    sys.stdout.write(f"\r  [{bar}] {percent}%")
    sys.stdout.flush()

def download_from_url(url, output_path):
    """Download dataset from a user-provided URL."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"  Downloading dataset from: {url}")
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print(f"\n  Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        return False

def extract_dataset(zip_path):
    """Extract the downloaded dataset."""
    print(f"  Extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print(f"  Extraction complete: {RAW_DATA_DIR}")
        return True
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="AUTSL Dataset Downloader")
    parser.add_argument("--url", help="Direct download URL for the AUTSL zip file (if available)")
    parser.add_argument("--metadata-only", action="store_true", help="Only download metadata")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  AUTSL (Turkish Sign Language) Downloader")
    print(f"{'='*60}")

    if not download_metadata():
        print("Continuing without fresh metadata...")

    if args.metadata_only:
        return

    if args.url:
        zip_path = os.path.join("datasets", "autsl_temp.zip")
        if download_from_url(args.url, zip_path):
            extract_dataset(zip_path)
            os.remove(zip_path) # Cleanup
    else:
        print("""
  [ATTENTION]
  AUTSL videos require registration at:
  https://chalearnlap.cvc.uab.cat/dataset/40/description/

  1. Register and download the dataset manually.
  2. Extract to: datasets/raw_videos/AUTSL/
  3. Or provide a direct link using:
     python training/data_pipeline/autsl_downloader.py --url <YOUR_LINK>
        """)

if __name__ == "__main__":
    main()
