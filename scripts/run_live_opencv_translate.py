"""
Run OpenCV live sign capture + translation.

Usage:
  python scripts/run_live_opencv_translate.py --camera 0
  python scripts/run_live_opencv_translate.py --list-cameras
  python scripts/run_live_opencv_translate.py --camera 1 --width 1280 --height 720 --fps 20 --flip
  python scripts/run_live_opencv_translate.py --camera 0 --tts --tts-cooldown 2.0
"""

import argparse
from vision_pipeline.live_capture import run_live, list_cameras


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--no-guides", action="store_true")
    parser.add_argument("--no-fps", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--tts", action="store_true")
    parser.add_argument("--tts-cooldown", type=float, default=2.0)
    args = parser.parse_args()
    if args.list_cameras:
        cams = list_cameras()
        print("Available cameras:", cams)
        return
    run_live(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps_limit=args.fps,
        window=args.window,
        flip=args.flip,
        show_fps=not args.no_fps,
        draw_guides=not args.no_guides,
        min_confidence=args.min_confidence,
        tts=args.tts,
        tts_cooldown=args.tts_cooldown,
    )


if __name__ == "__main__":
    main()
