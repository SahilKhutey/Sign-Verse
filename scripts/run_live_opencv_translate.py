"""
Run OpenCV live sign capture + translation.

Usage:
  python scripts/run_live_opencv_translate.py --camera 0
"""

import argparse
from vision_pipeline.live_capture import run_live


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run_live(camera_index=args.camera)


if __name__ == "__main__":
    main()
