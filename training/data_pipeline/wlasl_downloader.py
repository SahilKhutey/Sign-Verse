"""
WLASL Dataset Downloader

Downloads WLASL videos using their annotation JSON.
WLASL is the largest publicly available word-level ASL dataset.

Stats:
    21,083 videos | 2,000 sign classes | multiple signers

License: Attribution Non-Commercial 4.0 International

Usage:
    python training/data_pipeline/wlasl_downloader.py
    python training/data_pipeline/wlasl_downloader.py --limit 500
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
import concurrent.futures

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

ANNOTATION_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/"
    "start_kit/WLASL_v0.3.json"
)
OUTPUT_DIR = os.path.join("datasets", "raw_videos", "WLASL")
ANNOTATION_DIR = os.path.join("datasets", "metadata", "WLASL")


def download_annotation():
    """Download the WLASL annotation JSON."""
    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    ann_path = os.path.join(ANNOTATION_DIR, "WLASL_v0.3.json")

    if os.path.exists(ann_path):
        print(f"  Annotations already exist: {ann_path}")
        return ann_path

    print(f"  Downloading WLASL annotations...")
    try:
        urllib.request.urlretrieve(ANNOTATION_URL, ann_path)
        print(f"  Saved: {ann_path}")
        return ann_path
    except Exception as e:
        print(f"  Failed to download annotations: {e}")
        return None


def parse_annotations(ann_path, limit=None):
    """Parse WLASL JSON and return list of (video_id, url, gloss, split) tuples."""
    with open(ann_path) as f:
        data = json.load(f)

    entries = []
    for entry in data:
        gloss = entry["gloss"]
        for inst in entry.get("instances", []):
            video_id = inst.get("video_id", "")
            url = inst.get("url", "")
            split = inst.get("split", "train")
            frame_start = inst.get("frame_start", -1)
            frame_end = inst.get("frame_end", -1)
            if video_id and url:
                entries.append({
                    "video_id": video_id,
                    "url": url,
                    "gloss": gloss,
                    "split": split,
                    "frame_start": frame_start,
                    "frame_end": frame_end
                })
        if limit and len(entries) >= limit:
            break

    print(f"  Found {len(entries):,} video entries")
    return entries


def download_video(entry, output_dir, timeout=30):
    """Download a single video. Returns (success, video_id, reason)."""
    video_id = entry["video_id"]
    url = entry["url"]
    gloss = entry["gloss"]

    # Create class-level subdirectory
    class_dir = os.path.join(output_dir, gloss.upper().replace(" ", "_"))
    os.makedirs(class_dir, exist_ok=True)

    output_path = os.path.join(class_dir, f"{video_id}.mp4")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        return (True, video_id, "already_exists")

    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Research) SignVerse/1.0'
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()

        if len(data) < 1024:
            return (False, video_id, "too_small")

        with open(output_path, "wb") as f:
            f.write(data)

        return (True, video_id, "downloaded")

    except urllib.error.HTTPError as e:
        return (False, video_id, f"HTTP {e.code}")
    except Exception as e:
        return (False, video_id, str(e)[:50])


def download_all(entries, output_dir, max_workers=8):
    """Download all videos with parallel execution."""
    os.makedirs(output_dir, exist_ok=True)

    total = len(entries)
    success = 0
    failed = 0
    already = 0

    print(f"\n  Starting download of {total:,} videos ({max_workers} threads)...")
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_video, e, output_dir): e
            for e in entries
        }

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            ok, vid, reason = future.result()
            if ok and reason == "already_exists":
                already += 1
            elif ok:
                success += 1
            else:
                failed += 1

            # Progress
            if (i + 1) % 100 == 0 or (i + 1) == total:
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1)
                eta = (total - i - 1) / max(rate, 1)
                print(f"  [{i+1:>6}/{total}] ✓{success} ✗{failed} "
                      f"={already} | {rate:.1f}/s | ETA {eta/60:.1f}min")

    print(f"\n  Download complete:")
    print(f"    Downloaded:    {success:,}")
    print(f"    Failed:        {failed:,}")
    print(f"    Already had:   {already:,}")
    print(f"    Total time:    {(time.time()-t_start)/60:.1f} min")

    return success, failed


def main():
    parser = argparse.ArgumentParser(description="WLASL Dataset Downloader")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of videos (for testing)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel downloads")
    parser.add_argument("--split", choices=["train", "val", "test", "all"],
                        default="all", help="Which split to download")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  WLASL Dataset Downloader")
    print(f"  2,000 ASL signs | 21,083 videos")
    print(f"  License: CC Attribution Non-Commercial 4.0")
    print(f"{'='*60}")

    # Download annotations
    ann_path = download_annotation()
    if not ann_path:
        print("Cannot proceed without annotations.")
        return

    # Parse
    entries = parse_annotations(ann_path, limit=args.limit)

    # Filter by split
    if args.split != "all":
        entries = [e for e in entries if e["split"] == args.split]
        print(f"  Filtered to '{args.split}' split: {len(entries):,} videos")

    # Download
    download_all(entries, OUTPUT_DIR, max_workers=args.workers)


if __name__ == "__main__":
    main()
