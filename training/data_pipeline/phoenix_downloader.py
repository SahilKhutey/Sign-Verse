"""
RWTH-PHOENIX-Weather 2014T Dataset Downloader

Direct FTP download links discovered (no registration required for FTP access):
    2014T (39GB): https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz
    2014  (53GB): https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014.v3.tar.gz

License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
Registered with: gensoulslab@gmail.com

Dataset stats (2014T):
    Train:  7,096 sequences
    Dev:      519 sequences
    Test:     642 sequences
    Vocab:  1,200 signs

Usage:
    python training/data_pipeline/phoenix_downloader.py --version 2014T
    python training/data_pipeline/phoenix_downloader.py --version 2014
"""

import os
import sys
import urllib.request
import tarfile
import argparse

OUTPUT_DIR = os.path.join("datasets", "raw_videos", "RWTH_PHOENIX")

URLS = {
    "2014T": "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz",
    "2014":  "https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014.v3.tar.gz",
}

SIZES = {
    "2014T": "39 GB",
    "2014":  "53 GB",
}


def progress_hook(block_count, block_size, total_size):
    downloaded = block_count * block_size
    percent = min(int(downloaded * 100 / max(total_size, 1)), 100)
    mb = downloaded / 1e6
    total_mb = total_size / 1e6
    bar = "█" * (percent // 2) + "░" * (50 - percent // 2)
    sys.stdout.write(f"\r  [{bar}] {percent}% — {mb:.0f}/{total_mb:.0f} MB")
    sys.stdout.flush()


def download_phoenix(version="2014T"):
    """Download and extract RWTH-PHOENIX dataset."""
    url = URLS.get(version)
    size = SIZES.get(version, "?")

    if not url:
        print(f"Unknown version: {version}. Choose from: {list(URLS.keys())}")
        return False

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fname = url.split("/")[-1]
    tar_path = os.path.join(OUTPUT_DIR, fname)
    extract_dir = os.path.join(OUTPUT_DIR, version)

    print(f"\n{'='*60}")
    print(f"  RWTH-PHOENIX-Weather {version}")
    print(f"  Size: {size}")
    print(f"  URL:  {url}")
    print(f"  Dest: {tar_path}")
    print(f"{'='*60}")

    # Download
    if os.path.exists(tar_path):
        print(f"\n  Archive already downloaded: {tar_path}")
    else:
        print(f"\n  Downloading {size} archive...")
        print("  (This will take a while on slower connections)")
        try:
            urllib.request.urlretrieve(url, tar_path, reporthook=progress_hook)
            print(f"\n  Download complete!")
        except KeyboardInterrupt:
            print("\n  Download interrupted. Run again to resume (partial file).")
            return False
        except Exception as e:
            print(f"\n  Download failed: {e}")
            print(f"\n  If download fails, manually download from:")
            print(f"  {url}")
            print(f"  And extract to: {OUTPUT_DIR}/")
            return False

    # Extract
    if os.path.exists(extract_dir):
        print(f"\n  Already extracted to: {extract_dir}")
    else:
        print(f"\n  Extracting archive...")
        os.makedirs(extract_dir, exist_ok=True)
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                members = tar.getmembers()
                total = len(members)
                for i, member in enumerate(members):
                    tar.extract(member, path=extract_dir)
                    if (i + 1) % 1000 == 0:
                        print(f"  Extracted {i+1:,}/{total:,} files...")
            print(f"  Extraction complete: {extract_dir}")
        except Exception as e:
            print(f"  Extraction failed: {e}")
            return False

    # Print dataset structure
    print(f"\n  Dataset structure:")
    for split in ["train", "dev", "test"]:
        split_dir = os.path.join(extract_dir, "phoenix-2014-T.v3", "PHOENIX-2014-T-release-v3", split)
        if os.path.exists(split_dir):
            count = sum(1 for r, d, f in os.walk(split_dir) for _ in f)
            print(f"    {split:<8} {count:>6} files")

    print(f"\n  RWTH-PHOENIX {version} ready for preprocessing!")
    print(f"  Run: python training/data_pipeline/unified_preprocessor.py")
    return True


def main():
    parser = argparse.ArgumentParser(description="RWTH-PHOENIX Dataset Downloader")
    parser.add_argument("--version", choices=["2014T", "2014", "both"],
                        default="2014T")
    args = parser.parse_args()

    if args.version == "both":
        for v in ["2014T", "2014"]:
            download_phoenix(v)
    else:
        download_phoenix(args.version)


if __name__ == "__main__":
    main()
