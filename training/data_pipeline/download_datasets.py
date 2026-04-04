"""
Dataset Downloader — Downloads all publicly available
sign language datasets with progress tracking.

Supported datasets:
    - WLASL        (annotations JSON — free)
    - AUTSL        (request required)
    - RWTH-PHOENIX (request required)
    - How2Sign     (CC BY 4.0 — free)
    - MS-ASL       (request required)
    - LSA64        (free)

For datasets requiring institutional access,
the script prints direct instructions.

Usage:
    python download_datasets.py --datasets WLASL LSA64 How2Sign
    python download_datasets.py --all
    python download_datasets.py --list
"""

import os
import sys
import json
import argparse
import urllib.request
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from training.data_pipeline.dataset_registry import DATASETS, print_summary
from training.data_pipeline.autsl_downloader import download_metadata as download_autsl_metadata


RAW_DATA_DIR = os.path.join("datasets", "raw_videos")
ANNOTATION_DIR = os.path.join("datasets", "metadata")


def progress_hook(count, block_size, total_size):
    """Show download progress bar."""
    percent = min(int(count * block_size * 100 / max(total_size, 1)), 100)
    bar = "█" * (percent // 2) + "░" * (50 - percent // 2)
    sys.stdout.write(f"\r  [{bar}] {percent}%")
    sys.stdout.flush()


def download_file(url, output_path):
    """Download a file with progress display."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
        print(f"\n  Saved → {output_path}")
        return True
    except Exception as e:
        print(f"\n  Download failed: {e}")
        return False


def download_wlasl():
    """Download WLASL annotations (JSON). Videos need separate access."""
    print("\n[WLASL] Downloading annotations...")
    info = DATASETS["WLASL"]
    ann_url = info["annotation_url"]
    ann_path = os.path.join(ANNOTATION_DIR, "WLASL", "WLASL_v0.3.json")

    if download_file(ann_url, ann_path):
        with open(ann_path) as f:
            data = json.load(f)
        print(f"  Annotations loaded: {len(data)} signs")

    print("""
  [WLASL VIDEO ACCESS]
  Videos are hosted individually. To download:
    1. Visit: https://github.com/dxli94/WLASL
    2. Run: python start_kit/video_downloader.py
    3. Videos saved to: datasets/raw_videos/WLASL/
  """)


def download_lsa64():
    """LSA64 — Argentine Sign Language (64 words, 3200 videos)."""
    print("""
  [LSA64] Download Instructions:
    Dataset: 3,200 videos — 64 Argentine Sign Language words
    Direct link: https://mega.nz/folder/hTBTDSIL#9rDtKspGThlHY2iqGUgcvQ
    1. Download and extract to: datasets/raw_videos/LSA64/
    2. Directory structure expected:
         LSA64/001/001_001.mp4
         LSA64/001/001_002.mp4
    """)


def download_how2sign():
    """How2Sign — ASL sentence-level (CC BY 4.0, ~300GB)."""
    print("""
  [How2Sign] Download Instructions:
    Dataset: ~80,000 sentence-level ASL videos (CC BY 4.0)
    URL: https://how2sign.github.io/

    Steps:
    1. Visit: https://how2sign.github.io/
    2. Download the sign video files and annotation CSVs
    3. Extract to: datasets/raw_videos/How2Sign/
    4. Download annotations:
         how2sign_train.csv
         how2sign_val.csv
         how2sign_test.csv

    Note: ~300GB total — use a fast internet connection.
    """)


def download_rwth_phoenix():
    """RWTH-PHOENIX-Weather 2014T — requires institutional registration."""
    print("""
  [RWTH-PHOENIX-Weather 2014T] Access Instructions:
    Dataset: 45,760 samples | vocab: 1200 | language: DGS (~53GB)

    Requires institutional registration:
    1. Visit: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/
    2. Submit access request form
    3. After approval, download:
         phoenix-2014-T.v3.tar.gz
    4. Extract to: datasets/raw_videos/RWTH_PHOENIX/
    """)


def download_autsl():
    """AUTSL — Turkish Sign Language (automated metadata + instructions)."""
    print("\n[AUTSL] Processing metadata...")
    download_autsl_metadata()
    
    print("""
  [AUTSL] Video Access Instructions:
    Dataset: 38,336 samples | vocab: 226 | language: TSL

    Requires registration at:
    https://chalearnlap.cvc.uab.cat/dataset/40/description/

    After download, extract to: datasets/raw_videos/AUTSL/
    
    Or run:
    python training/data_pipeline/autsl_downloader.py --url <YOUR_LINK>
    """)


def download_ms_asl():
    """MS-ASL — Microsoft ASL (requires request)."""
    print("""
  [MS-ASL] Access Instructions:
    Dataset: 25,000 samples | vocab: 1000 | language: ASL

    Request access at:
    https://www.microsoft.com/en-us/research/project/ms-asl/

    After approval, extract to: datasets/raw_videos/MS_ASL/
    """)


DOWNLOAD_HANDLERS = {
    "WLASL": download_wlasl,
    "LSA64": download_lsa64,
    "How2Sign": download_how2sign,
    "RWTH_PHOENIX": download_rwth_phoenix,
    "AUTSL": download_autsl,
    "MS_ASL": download_ms_asl,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download public sign language datasets"
    )
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()),
                        help="Specific datasets to download")
    parser.add_argument("--all", action="store_true",
                        help="Download all datasets")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets")
    parser.add_argument("--free-only", action="store_true",
                        help="Only download datasets not requiring requests")
    args = parser.parse_args()

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(ANNOTATION_DIR, exist_ok=True)

    if args.list:
        print_summary()
        return

    targets = []
    if args.all:
        targets = list(DATASETS.keys())
    elif args.free_only:
        targets = [k for k, v in DATASETS.items() if not v["requires_request"]]
    elif args.datasets:
        targets = args.datasets
    else:
        # Default: show instructions for all
        targets = list(DATASETS.keys())

    print(f"\n{'='*60}")
    print(f"  SignVerse Dataset Downloader")
    print(f"  Datasets: {', '.join(targets)}")
    print(f"  Total samples: {sum(DATASETS[d]['samples'] for d in targets):,}")
    print(f"{'='*60}")

    for name in targets:
        print(f"\n{'─'*40}")
        print(f"  Processing: {name}")
        print(f"  Samples: {DATASETS[name]['samples']:,} | "
              f"Vocab: {DATASETS[name]['vocab_size']:,} | "
              f"Lang: {DATASETS[name]['language']}")
        print(f"{'─'*40}")

        handler = DOWNLOAD_HANDLERS.get(name)
        if handler:
            handler()
        else:
            print(f"  No handler for {name}")

    print(f"\n{'='*60}")
    print("  Setup complete. After downloading:")
    print("  Run: python training/data_pipeline/unified_preprocessor.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
